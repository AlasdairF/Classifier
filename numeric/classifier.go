package classifier

import (
 "math"
 "math/rand"
 "errors"
 "fmt"
 "os"
 "encoding/gob"
 "compress/gzip"
)

// --------------- CONSTANTS ---------------

/*

The number of ensembles can be changed here from 20 to any other number.
It works best on 20, that's why it's hardcoded.

*/
const number_of_ensembles = 20


// --------------- STRUCTS ---------------

type Trainer struct {
Classifier // inherits Classifier struct
testDocs [][][]int64
trainingTokens [][]int64
Category_index map[string]int // Category_index can be useful as it contains a map of cat => index, where index is the slice index of the category in Classifier.Categories
category_ensemble_index map[string][number_of_ensembles]int
num_test_docs int
ensembleContent [][]word
ensembled bool
}

type Classifier struct {
Categories []string
ClassifierVar map[int64][]Scorer
}

type word struct {
tok int64
score float32
}

// You can ignore this. Scorer must be able to be exported (begin with a capital) so it can be accessed by the gob functionality used for Load & Save.
type Scorer struct {
Category uint16
Score float32
}

// --------------- FUNCTIONS ---------------

// randomList is a helper function to generate random lists of integers for the ensemble function. It does not need to be seeded since it is good for the random numbers to be the same for the same content.
func randomList(num int, wanted int) []int {
	output := make([]int,wanted)
	used := make([]bool,num)
	for got:=0; got<wanted; { // while got<wanted
		n := rand.Intn(num) // generate random number
		if used[n]==false { // check if its used already
			output[got]=n
			used[n]=true
			got++
		}
	}
	return output
}

// DefineCategories creates categories from a slice of categories names as strings.
func (t *Trainer) DefineCategories(categories []string) {
	// Reset & init
	t.category_ensemble_index = make(map[string][number_of_ensembles]int)
	t.Category_index = make(map[string]int)
	// Now generate forward and reverse index for categories <-> ensembles <-> indices
	t.Categories=categories
	current := 0
	for i,category := range categories {
		t.Category_index[category]=i
		var temp [number_of_ensembles]int
		for i2:=0; i2<number_of_ensembles; i2++ {
			temp[i2]=current
			current++
		}
		t.category_ensemble_index[category]=temp
	}
	t.testDocs = make([][][]int64,len(categories))
	t.trainingTokens = make([][]int64,len(categories))
}

// AddTrainingDoc adds a training document to the classifier.
func (t *Trainer) AddTrainingDoc(category string, tokens []int64) error {
	t.ensembled=false // Needs to be ensembled whenever a training doc is added
	// Check to see if category exists already, if it doesn't then add it
	indx,ok := t.Category_index[category]
	if !ok {
		return errors.New(`AddTrainingDoc: Category '`+category+`' not defined`)
	}
	// Check trainingTokens[indx] capacity and grow if necessary
	l := len(t.trainingTokens[indx])
	n := l+len(tokens)
	if (n>cap(t.trainingTokens[indx])) {
		newSlice := make([]int64, n, n*2)
        copy(newSlice, t.trainingTokens[indx])
        t.trainingTokens[indx] = newSlice
	} else {
		t.trainingTokens[indx] = t.trainingTokens[indx][0:n]
	}
	// Now add the new tokens
	for i,i2 := l,0; i<n; i++ {
		t.trainingTokens[indx][i]=tokens[i2]
		i2++
	}
	return nil
}

// AddTestDoc adds a document for testing under the Test function.
func (t *Trainer) AddTestDoc(category string, tokens []int64) error {
	// Check to see if category exists already, if it doesn't then add it
	indx,ok := t.Category_index[category]
	if !ok {
		return errors.New(`AddTestDoc: Category '`+category+`' not defined`)
	}
	// Check capacity and grow if necessary
	l := len(t.testDocs[indx])
	if (l==cap(t.testDocs[indx])) {
		newSlice := make([][]int64, l+1, (l+1)*2)
        copy(newSlice, t.testDocs[indx])
        t.testDocs[indx] = newSlice
	} else {
		t.testDocs[indx] = t.testDocs[indx][0:l+1]
	}
	// Add training doc to this category and increase total number of training docs
	t.testDocs[indx][l]=tokens
	t.num_test_docs++
	return nil
}

// ensemble does most of the calculations and pruning for the classifier, which is then finished off by Create.
func (t *Trainer) ensemble() {
	// Initialize
	total := uint64(0)
	nlist := make([]int,len(t.Categories)*number_of_ensembles)
	tokmap := make([]map[int64]uint,len(t.Categories)*number_of_ensembles)
	bigmap := make(map[int64]uint)
	// Loop through all categories of training docs
	for indx,cat := range t.Categories {
		// Generate 20x ensembles of 50% tokens
		num_tokens := len(t.trainingTokens[indx])
		per_ensemble := (num_tokens+1)/2
		for i:=0; i<number_of_ensembles; i++ {
			ensembleindx := t.category_ensemble_index[cat][i]
			tokloop := randomList(num_tokens,per_ensemble) // select 50% random sampling for this category
			nlist[ensembleindx]=per_ensemble
			total += uint64(per_ensemble)
			tokmap[ensembleindx] = make(map[int64]uint)
			for i2:=0; i2<per_ensemble; i2++ {
				tok := t.trainingTokens[indx][tokloop[i2]]
				tokmap[ensembleindx][tok]++
				bigmap[tok]++
			}
		}
	}
	// And add to the overall counts
	ensembleTokAvg := make(map[int64]float32)
	l := float32(total)
	for tok,count := range bigmap {
		ensembleTokAvg[tok]=float32(count)/l
	}
	// Now prune ensembleContent to remove all that are less than avg and therefore useless
	t.ensembleContent = make([][]word,len(t.Categories)*number_of_ensembles)
	for _,cat := range t.Categories { // loop through categories
		for i:=0; i<number_of_ensembles; i++ { // loop through ensemble categories
			ensembleindx := t.category_ensemble_index[cat][i] // get the index for this ensemble category
			ensembleContent := make([]word,len(tokmap[ensembleindx]))
			good := 0
			for tok,count := range tokmap[ensembleindx] {
				if count>1 {
					if avg := float32(count)/float32(nlist[ensembleindx]); avg>ensembleTokAvg[tok] {
						score := avg/ensembleTokAvg[tok]
						ensembleContent[good] = word{tok,score}
						good++
					}
				}
			}
			// And save the pruned ensembleContent into the struct
			t.ensembleContent[ensembleindx] = make([]word,good)
			copy(t.ensembleContent[ensembleindx],ensembleContent[0:good])
		}
	}
	return
}

// Create builds the classifier using the two variables allowance & maxscore. Set allowance & maxscore to 0 for no limits.
func (t *Trainer) Create(allowance float32, maxscore float32) {
	// First run ensemble if it hasn't been run already
	if t.ensembled==false {
		t.ensemble()
		t.ensembled=true
	}
	// Now build the classifier
	t.ClassifierVar = make(map[int64][]Scorer)
	for indx,cat := range t.Categories { // loop through categories
		tally := make(map[int64]float32) // create tally for scores from this category
		for i:=0; i<number_of_ensembles; i++ { // loop through ensemble categories
			ensembleindx := t.category_ensemble_index[cat][i] // get the index for this ensemble category
			l := len(t.ensembleContent[ensembleindx]) // get the number of tokens in this ensemble category
			for i2:=0; i2<l; i2++ { // loop through all the tokens in this ensemble category
				score := t.ensembleContent[ensembleindx][i2].score // calculate the score of this token
				if score>=allowance { // If the score is greater than the allowance
					if maxscore>0 && score>maxscore { // if score is greater than the maximum allowed score for one token then reduce it to the maximum
						score = maxscore
					}
					tally[t.ensembleContent[ensembleindx][i2].tok]+=score // Add token and score to the tally for this category
					}
				}
			}
		// Enter tallys into classifier
		indx16 := uint16(indx)
		for tok,score := range tally {
			scorelog := float32(math.Log(float64(score)))
			if old,ok := t.ClassifierVar[tok]; ok {
				temp := len(old)
				newone := make([]Scorer,temp+1)
				copy(newone,old)
				newone[temp]=Scorer{indx16,scorelog}
				t.ClassifierVar[tok]=newone
			} else {
				newone := make([]Scorer,1)
				newone[0]=Scorer{indx16,scorelog}
				t.ClassifierVar[tok]=newone			
			}
		}
	}
}

// Classify classifies tokens and returns a slice of float32 where each index is the same as the index for the category name in classifier.Categories, which is the same as the []string of categories originally past to DefineCategories.
func (t *Classifier) Classify(tokens []int64) []float64 {
	scoreboard := make([]float64,len(t.Categories))
	for _,tok := range tokens {
		if rules,ok := t.ClassifierVar[tok]; ok {
			for i:=0; i<len(rules); i++ {
				scoreboard[rules[i].Category]+=float64(rules[i].Score)
			}
		}
	}
	return scoreboard
}

// ClassifySimple is a wrapper for Classify, it returns the name of the best category as a string, and the score of the best category as float32.
func (t *Classifier) ClassifySimple(tokens []int64) (string, float64) {
	scoreboard := t.Classify(tokens)
	var bestscore float64
	var bestcat int
	for cat,score := range scoreboard {
		if score>bestscore {
			bestscore=score
			bestcat=cat
			}
		}
	return t.Categories[bestcat], bestscore
}

// Test tries 2,401 different combinations of allowance & maxscore then returns the values of allowance & maxscore which performs the best. Test requires an argument of true or false for verbose, if true Test will print all results to Stdout. 
func (t *Trainer) Test(verbose bool) (float32, float32, error) {
	// Check there are test files
	if t.num_test_docs==0 {
		err := errors.New(`Test: Add test files`)
		return 0, 0, err
	}
	// Set some variables
	var bestaccuracy float32
	var bestallowance float32
	var bestmaxscore float32
	// auto is the list of numbers to try for allowance and maxscore
	var auto_allowance = [...]float32{0,1.05,1.1,1.15,1.2,1.25,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.5,3,4,5,6,7,8,9,10,15,20,25,30,40,50,75,100,150,200,300,400,500,600,700,800,900,1000,1500,2000,3000,4000,5000,10000,20000,50000,100000,1000000}
	var auto_maxscore = [...]float32{0,10000000,1000000,100000,50000,20000,10000,5000,4000,3000,2000,1500,1200,1000,900,800,700,600,550,500,475,450,425,400,375,350,325,300,275,250,225,200,150,100,75,50,40,30,25,20,15,10,8,6,4,2}
	for _,allowance := range auto_allowance { // loop through auto for allowance
		for _,maxscore := range auto_maxscore { // loop through auto for maxscore
			t.Create(allowance,maxscore) // build the classifier for allowance & maxscore
			correct := 0
			// Count the number of correct results from testDocs under this classifier
			for indx,cat := range t.Categories {
				for i:=0; i<len(t.testDocs[indx]); i++ {
					if temp,_ := t.ClassifySimple(t.testDocs[indx][i]); temp==cat {
						correct++
					}
				}
			}
			// And the accuracy is:
			accuracy := float32(correct)/float32(t.num_test_docs)
			if verbose {
				fmt.Printf("allowance %g, maxscore %g = %f (%d correct)\n",allowance,maxscore,accuracy,correct)
			}
			// Save the best accuracy
			if accuracy>bestaccuracy {
				bestaccuracy=accuracy
				bestallowance=allowance
				bestmaxscore=maxscore
			}
		}
	}
	if verbose {
		fmt.Println(`BEST RESULT`)
		fmt.Printf("allowance %g, maxscore %g = %f\n",bestallowance,bestmaxscore,bestaccuracy)
	}
	return bestallowance, bestmaxscore, nil
}

// Loads a classifier from a file previously saved with Save.
func (t *Classifier) Load(filename string) error {
	// Open file for reading
	fi, err := os.Open(filename)
	if err !=nil {
		return err
	}
	defer fi.Close()
	// Attached gzip reader
    fz, err := gzip.NewReader(fi)
	if err !=nil {
		return err
	}
	defer fz.Close()
	// Pull with the gob decoder
	decoder := gob.NewDecoder(fz)
	err = decoder.Decode(&t)
	if err !=nil {
		return err
	}
	return nil
}

// Saves classifier last created with Create to a file.
func (t *Trainer) Save(filename string) error {
	// Open file for writing
	fi, err := os.Create(filename)
	if err !=nil {
		return err
	}
	defer fi.Close()
	// Attach gzip writer
	fz := gzip.NewWriter(fi)
	defer fz.Close()
	// Push from the gob encoder
	encoder := gob.NewEncoder(fz)
	err = encoder.Encode(t.Classifier)
	if err !=nil {
		return err
	}
	return nil
}
