package classifier

import (
 "github.com/AlasdairF/Custom"
 "os"
 "math"
 "math/rand"
 "errors"
 "fmt"
)

/*

NOTE:
	Maximum byte length of any token or category name is 255 bytes (UTF8).
	Maximum number of categories is 65,536.

*/

// --------------- CONSTANTS ---------------

/*

The number of ensembles can be changed here from 20 to any other number.
It works best on 20, that's why it's hardcoded.

*/
const number_of_ensembles = 20


// --------------- STRUCTS ---------------

type Trainer struct {
Classifier // inherits Classifier struct
TestDocs [][][]uint32
TrainingTokens [][]uint32
Category_index map[string]int // Category_index can be useful as it contains a map of cat => index, where index is the slice index of the category in Classifier.Categories
ensembleContent [][]word
ensembled bool
}

type Classifier struct {
Categories []string
rules map[uint32][]scorer
}

type word struct {
 tok uint32
 score float32
}

type scorer struct {
 category uint16
 score float32
}

// --------------- FUNCTIONS ---------------

// randomList is a helper function to generate random lists of integers for the ensemble function. It does not need to be seeded since it is good for the random numbers to be the same for the same content.
func randomList(num int, wanted int) []int {
	output := make([]int, wanted)
	used := make([]bool, num)
	var n int
	for got:=0; got<wanted; { // while got<wanted
		n = rand.Intn(num) // generate random number
		if !used[n] { // check if its used already
			output[got] = n
			used[n] = true
			got++
		}
	}
	return output
}

// DefineCategories creates categories from a slice of categories names as strings.
func (t *Trainer) DefineCategories(categories []string) {
	// Reset & init
	t.Category_index = make(map[string]int)
	// Now generate forward and reverse index for categories <-> ensembles <-> indices
	t.Categories = categories
	for i, category := range categories {
		t.Category_index[category] = i
	}
	t.TestDocs = make([][][]uint32, len(categories))
	t.TrainingTokens = make([][]uint32, len(categories))
}

// AddTrainingDoc adds a training document to the classifier.
func (t *Trainer) AddTrainingDoc(category string, tokens []uint32) error {
	t.ensembled = false // Needs to be ensembled whenever a training doc is added
	// Check to see if category exists already, if it doesn't then add it
	indx, ok := t.Category_index[category]
	if !ok {
		return errors.New(`AddTrainingDoc: Category '` + category + `' not defined`)
	}
	// Add tokens
	t.TrainingTokens[indx] = append(t.TrainingTokens[indx], tokens...)
	return nil
}

// AddTestDoc adds a document for testing under the Test function.
func (t *Trainer) AddTestDoc(category string, tokens []uint32) error {
	// Check to see if category exists already, if it doesn't then add it
	indx, ok := t.Category_index[category]
	if !ok {
		return errors.New(`AddTestDoc: Category '` + category + `' not defined`)
	}
	// Check capacity and grow if necessary
	t.TestDocs[indx] = append(t.TestDocs[indx], tokens)
	return nil
}

// ensemble does most of the calculations and pruning for the classifier, which is then finished off by Create.
func (t *Trainer) ensemble() {
	// Initialize
	nlist := make([]int, len(t.Categories) * number_of_ensembles)
	tokmap := make([]map[uint32]uint, len(t.Categories) * number_of_ensembles)
	bigmap := make(map[uint32]uint)
	var i, i2, indx, ensembleindx, num_tokens, per_ensemble int
	var total uint32
	var tokloop []int
	var tok uint32
	numcats := len(t.Categories)
	// Loop through all categories of training docs
	for indx=0; indx<numcats; indx++ {
		// Generate 20x ensembles of 50% tokens
		num_tokens = len(t.TrainingTokens[indx])
		per_ensemble = (num_tokens+1)/2
		for i=0; i<number_of_ensembles; i++ {
			tokloop = randomList(num_tokens, per_ensemble) // select 50% random sampling for this category
			nlist[ensembleindx] = per_ensemble
			total += uint32(per_ensemble)
			tokmap[ensembleindx] = make(map[uint32]uint)
			for i2=0; i2<per_ensemble; i2++ {
				tok = t.TrainingTokens[indx][tokloop[i2]]
				tokmap[ensembleindx][tok]++
				bigmap[tok]++
			}
			ensembleindx++
		}
	}
	// And add to the overall counts
	ensembleTokAvg := make(map[uint32]float32)
	avg := float32(total)
	for tok, count := range bigmap {
		ensembleTokAvg[tok] = float32(count) / avg
	}
	var count uint
	var l int
	// Now prune ensembleContent to remove all that are less than avg and therefore useless
	t.ensembleContent = make([][]word, len(t.Categories) * number_of_ensembles)
	var ensembleContent []word
	ensembleindx = 0
	for indx=0; indx<numcats; indx++ { // loop through categories
		for i=0; i<number_of_ensembles; i++ { // loop through ensemble categories
			l = len(tokmap[ensembleindx])
			if l > len(ensembleContent) {
				ensembleContent = make([]word, l)
			}
			i2 = 0
			for tok, count = range tokmap[ensembleindx] {
				if count > 1 {
					if avg = float32(count) / float32(nlist[ensembleindx]); avg > ensembleTokAvg[tok] {
						ensembleContent[i2] = word{tok, avg / ensembleTokAvg[tok]}
						i2++
					}
				}
			}
			// And save the pruned ensembleContent into the struct
			t.ensembleContent[ensembleindx] = make([]word, i2)
			copy(t.ensembleContent[ensembleindx], ensembleContent[0:i2])
			ensembleindx++
		}
	}
	return
}

// Create builds the classifier using the two variables allowance & maxscore. Set allowance & maxscore to 0 for no limits.
func (t *Trainer) Create(allowance float32, maxscore float32) {
	// First run ensemble if it hasn't been run already
	if !t.ensembled {
		t.ensemble()
		t.ensembled = true
	}
	// Now build the classifier
	var i, i2, ensembleindx, l int
	var indx16 uint16
	var scorelog, score float32
	var old []scorer
	var ok bool
	var tok uint32
	t.rules = make(map[uint32][]scorer)
	for indx, _ := range t.Categories { // loop through categories
		tally := make(map[uint32]float32) // create tally for scores from this category
		for i=0; i<number_of_ensembles; i++ { // loop through ensemble categories
			ensembleindx = (indx * number_of_ensembles) + i // get the index for this ensemble category
			l = len(t.ensembleContent[ensembleindx]) // get the number of tokens in this ensemble category
			for i2=0; i2<l; i2++ { // loop through all the tokens in this ensemble category
				score = t.ensembleContent[ensembleindx][i2].score // calculate the score of this token
				if score >= allowance { // If the score is greater than the allowance
					if maxscore > 0 && score > maxscore { // if score is greater than the maximum allowed score for one token then reduce it to the maximum
						score = maxscore
					}
					tally[t.ensembleContent[ensembleindx][i2].tok] += score // Add token and score to the tally for this category
					}
				}
			}
		// Enter tallys into classifier
		indx16 = uint16(indx)
		for tok, score = range tally {
			scorelog = float32(math.Log(float64(score)))
			if old, ok = t.rules[tok]; ok {
				i2 = len(old)
				newone := make([]scorer, i2 + 1)
				copy(newone, old)
				newone[i2] = scorer{indx16, scorelog}
				t.rules[tok] = newone
			} else {
				t.rules[tok] = []scorer{scorer{indx16, scorelog}}	
			}
		}
	}
}

// Classify classifies tokens and returns a slice of float32 where each index is the same as the index for the category name in classifier.Categories, which is the same as the []string of categories originally past to DefineCategories.
func (t *Classifier) Classify(tokens []uint32) []float64 {
	var tok uint32
	var ok bool
	var rules []scorer
	var obj scorer
	scoreboard := make([]float64, len(t.Categories))
	for _, tok = range tokens {
		if rules, ok = t.rules[tok]; ok {
			for _, obj = range rules {
				scoreboard[obj.category] += float64(obj.score)
			}
		}
	}
	return scoreboard
}

// ClassifySimple is a wrapper for Classify, it returns the name of the best category as a string, and the score of the best category as float32.
func (t *Classifier) ClassifySimple(tokens []uint32) (string, float64) {
	scoreboard := t.Classify(tokens)
	var bestscore float64
	var bestcat int
	for cat, score := range scoreboard {
		if score > bestscore {
			bestscore = score
			bestcat = cat
		}
	}
	return t.Categories[bestcat], bestscore
}

// Test tries 2,401 different combinations of allowance & maxscore then returns the values of allowance & maxscore which performs the best. Test requires an argument of true or false for verbose, if true Test will print all results to Stdout. 
func (t *Trainer) Test(verbose bool) (float32, float32, error) {
	// Check there are test files
	num_test_docs := len(t.TestDocs)
	if num_test_docs == 0 {
		return 0, 0, errors.New(`Test: Add test files`)
	}
	// Set some variables
	var bestaccuracy, bestallowance, bestmaxscore, accuracy, allowance, maxscore float32
	var i, indx, correct, l int
	var cat, compare string
	// auto is the list of numbers to try for allowance and maxscore
	var auto_allowance = [...]float32{0,1.05,1.1,1.15,1.2,1.25,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.5,3,4,5,6,7,8,9,10,15,20,25,30,40,50,75,100,150,200,300,400,500,600,700,800,900,1000,1500,2000,3000,4000,5000,10000,20000,50000,100000,1000000}
	var auto_maxscore = [...]float32{0,10000000,1000000,100000,50000,20000,10000,5000,4000,3000,2000,1500,1200,1000,900,800,700,600,550,500,475,450,425,400,375,350,325,300,275,250,225,200,150,100,75,50,40,30,25,20,15,10,8,6,4,2}
	for _, allowance = range auto_allowance { // loop through auto for allowance
		for _, maxscore = range auto_maxscore { // loop through auto for maxscore
			t.Create(allowance, maxscore) // build the classifier for allowance & maxscore
			correct = 0
			// Count the number of correct results from TestDocs under this classifier
			for indx, cat = range t.Categories {
				l = len(t.TestDocs[indx])
				for i=0; i<l; i++ {
					if compare, _ = t.ClassifySimple(t.TestDocs[indx][i]); compare == cat {
						correct++
					}
				}
			}
			// And the accuracy is:
			accuracy = float32(correct)/float32(num_test_docs)
			if verbose {
				fmt.Printf("allowance %g, maxscore %g = %f (%d correct)\n", allowance, maxscore, accuracy, correct)
			}
			// Save the best accuracy
			if accuracy > bestaccuracy {
				bestaccuracy = accuracy
				bestallowance = allowance
				bestmaxscore = maxscore
			}
		}
	}
	if verbose {
		fmt.Println(`BEST RESULT`)
		fmt.Printf("allowance %g, maxscore %g = %f\n", bestallowance, bestmaxscore, bestaccuracy)
	}
	return bestallowance, bestmaxscore, nil
}

func MustLoad(filename string) *Classifier {
	t, err := Load(filename)
	if err != nil {
		panic(err)
	}
	return t
}

// Loads a classifier from a file previously saved with Save.
func Load(filename string) (*Classifier, error) {
	// Open file for reading
	fi, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer fi.Close()

	// Attach reader
	r, err := custom.NewReader(fi, 20480)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	
	var i uint16
	numcats := r.Read16()
	categories := make([]string, numcats)
	for i=0; i<numcats; i++ {
		categories[i] = r.ReadString8()
	}
	numrules := r.Read32()
	
	rules := make(map[uint32][]scorer)
	var i2 uint32
	var score float32
	var id, n uint16
	var tok uint32
	if numcats < 256 {
		for i2=0; i2<numrules; i2++ {
			tok = r.Read32()
			n = uint16(r.Read8())
			lst := make([]scorer, n)
			for i=0; i<n; i++ {
				id = uint16(r.Read8())
				score = r.ReadFloat32()
				lst[i] = scorer{id, score}
			}
			rules[tok] = lst
		}
	} else {
		for i2=0; i2<numrules; i2++ {
			tok = r.Read32()
			n = r.Read16()
			lst := make([]scorer, n)
			for i=0; i<n; i++ {
				id = r.Read16()
				score = r.ReadFloat32()
				lst[i] = scorer{id, score}
			}
			rules[tok] = lst
		}
	}
	
	// Make sure we're at the end and the checksum is OK
	err = r.EOF()
	if err != nil {
		return nil, errors.New(`Not a valid uint32-based classifier file.`)
	}
	
	// Return the new object
	t := new(Classifier)
	t.Categories = categories
	t.rules = rules
	return t, nil
}

func (t *Trainer) Save(filename string) error {
	// Open file for writing
	fi, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer fi.Close()
	
	w := custom.NewWriter(fi)
	defer w.Close()
	
	var tok uint32
	var lst []scorer
	var res scorer
	numcats := uint16(len(t.Categories))
	
	w.Write16(numcats)
	for _, cat := range t.Categories {
		w.WriteString8(cat)
	}
	w.Write32(uint32(len(t.rules)))

	if numcats < 256 {
		for tok, lst = range t.rules {
			w.Write32(tok)
			w.Write8(uint8(len(lst)))
			for _, res = range lst {
				w.Write8(uint8(res.category))
				w.WriteFloat32(res.score)
			}
		}
	} else {
		for tok, lst = range t.rules {
			w.Write32(tok)
			w.Write16(uint16(len(lst)))
			for _, res = range lst {
				w.Write16(res.category)
				w.WriteFloat32(res.score)
			}
		}
	}
	return nil
}
