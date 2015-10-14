package classifier

import (
 "github.com/AlasdairF/BinSearch"
 "github.com/AlasdairF/Custom"
 "os"
 "math"
 "math/rand"
 "errors"
 "fmt"
)

/*

NOTE:
	Maximum byte length of any token or category name is 65 bytes (UTF8).
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
 testDocs [][]binsearch.CounterBytes
 numTestDocs int
 trainingTokens [][][]byte
 categoryIndex binsearch.KeyValBytes
 ensembleContent [][]word
 ensembled bool
}

type Classifier struct {
 Categories [][]byte
 rules binsearch.KeyBytes
 res [][]scorer
}

type word struct {
 tok []byte
 score int
}

type scorer struct {
 category uint16
 score uint64
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
func (t *Trainer) DefineCategories(categories [][]byte) error {
	// Now generate forward and reverse index for categories <-> ensembles <-> indices
	t.Categories = categories
	for i, category := range categories {
		if t.categoryIndex.AddUnsorted(category, i) != nil {
			return errors.New(`Category name must be no more than 64 bytes.`)
		}
	}
	t.categoryIndex.Build()
	t.testDocs = make([][]binsearch.CounterBytes, len(categories))
	t.trainingTokens = make([][][]byte, len(categories))
	return nil
}

// AddTrainingDoc adds a training document to the classifier.
func (t *Trainer) AddTrainingDoc(category []byte, tokens [][]byte) error {
	t.ensembled = false // Needs to be ensembled whenever a training doc is added
	// Check to see if category exists already, if it doesn't then add it
	indx, ok := t.categoryIndex.Find(category)
	if !ok {
		return errors.New(`AddTrainingDoc: Category '` + string(category) + `' not defined`)
	}
	// Add tokens
	t.trainingTokens[indx] = append(t.trainingTokens[indx], tokens...)
	return nil
}

// AddTestDoc adds a document for testing under the Test function.
func (t *Trainer) AddTestDoc(category []byte, tokens [][]byte) error {
	// Check to see if category exists already, if it doesn't then add it
	indx, ok := t.categoryIndex.Find(category)
	if !ok {
		return errors.New(`AddTestDoc: Category '` + string(category) + `' not defined`)
	}
	// Check capacity and grow if necessary
	t.testDocs[indx] = append(t.testDocs[indx], binsearch.CounterBytes{})
	obj := &t.testDocs[indx][len(t.testDocs[indx])-1]
	for _, word := range tokens {
		obj.Add(word, 1)
	}
	obj.Build()
	
	t.numTestDocs++
	return nil
}

// ensemble does most of the calculations and pruning for the classifier, which is then finished off by Create.
func (t *Trainer) ensemble() {
	// Initialize
	nlist := make([]int, len(t.Categories) * number_of_ensembles)
	tokmap := make([]binsearch.CounterBytes, len(t.Categories) * number_of_ensembles)
	ensembleTokAvg := new(binsearch.CounterBytes)
	var i, i2, indx, ensembleindx, num_tokens, per_ensemble, total int
	var tokloop []int
	var tok []byte
	numcats := len(t.Categories)
	// Loop through all categories of training docs
	for indx=0; indx<numcats; indx++ {
		// Generate 20x ensembles of 50% tokens
		num_tokens = len(t.trainingTokens[indx])
		per_ensemble = (num_tokens + 1) / 2
		for i=0; i<number_of_ensembles; i++ {
			tokloop = randomList(num_tokens, per_ensemble) // select 50% random sampling for this category
			nlist[ensembleindx] = per_ensemble
			total += per_ensemble
			//tokmap[ensembleindx] = make(map[string]uint)
			for i2=0; i2<per_ensemble; i2++ {
				tok = t.trainingTokens[indx][tokloop[i2]]
				tokmap[ensembleindx].Add(tok, 1)
				ensembleTokAvg.Add(tok, 1)
			}
			tokmap[ensembleindx].Build()
			ensembleindx++
		}
	}
	ensembleTokAvg.Build()
	
	// Calculate frequency for each token across all categories, multiplied by 10,000,000 so it will fit into an int instead of requiring a float
	ensembleTokAvg.UpdateAll(func(v int) int {return (v * 10000000) / total})
	
	var count, l, av, v int
	var eof, ok bool
	// Now prune ensembleContent to remove all that are less than avg and therefore useless
	t.ensembleContent = make([][]word, len(t.Categories) * number_of_ensembles)
	var ensembleContent []word
	ensembleindx = 0
	for indx=0; indx<numcats; indx++ { // loop through categories
		for i=0; i<number_of_ensembles; i++ { // loop through ensemble categories
			// Check the size of the temporary working array
			l = tokmap[ensembleindx].Len()
			if l > len(ensembleContent) {
				ensembleContent = make([]word, l)
			}
			// Loop through all tokens in this ensemble
			i2 = 0
			if tokmap[ensembleindx].Reset() {
				for eof = false; !eof; {
					tok, count, eof = tokmap[ensembleindx].Next() // get the next one
					if count >= 2 { // there must be at least 2 occurances of this token in this ensemble
						av = (count * 10000000) / nlist[ensembleindx] // Calculatate frequency for this token within this ensemble
						v, ok = ensembleTokAvg.Find(tok) // what's the average for this token overall?
						if av > v && ok { // if this token frequency in this ensemble is greater than average for all categories and ensembles
							ensembleContent[i2] = word{tok, (av * 1000) / v} // the result is the percentage over the average that this tokens occurs in this ensemble, multiplied by 1000 so it fits. It will always be >1 (or in this case >1000)
							i2++
						}
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
	allowanceint, maxscoreint := int(allowance * 1000), int(maxscore * 1000)
	var i, ensembleindx, score int
	var indx16 uint16
	var scorelog uint64
	var eof bool
	var tok []byte
	
	// First loop through and calculate exactly how many words will be included in the classifier
	dupfinder := new(binsearch.CounterBytes) // create tally for scores from this category
	for indx, _ := range t.Categories { // loop through categories
		for i=0; i<number_of_ensembles; i++ { // loop through ensemble categories
			for _, obj := range t.ensembleContent[(indx * number_of_ensembles) + i] {
				if obj.score >= allowanceint { // If the score is greater than the allowance
					dupfinder.Add(obj.tok, 0)
				}
			}
		}
	}
	
	// Convert the tally into a KeyBytes structure, which is the dictionary of tokens
	dupfinder.Build()
	rules := dupfinder.KeyBytes()
	dupfinder = nil
	res := make([][]scorer, rules.Len())
	
	// Now calculate the score for each dictionary token for each category
	for indx, _ := range t.Categories { // loop through categories
		tally := new(binsearch.CounterBytes) // create tally for scores from this category
		for i=0; i<number_of_ensembles; i++ { // loop through ensemble categories
			ensembleindx = (indx * number_of_ensembles) + i // get the index for this ensemble category
			for _, obj := range t.ensembleContent[ensembleindx] {
				if obj.score >= allowanceint { // If the score is greater than the allowance
					if maxscoreint > 0 && obj.score > maxscoreint { // if score is greater than the maximum allowed score for one token then reduce it to the maximum
						tally.Add(obj.tok, maxscoreint)
					} else {
						tally.Add(obj.tok, obj.score)
					}
				}
			}
		}
		tally.Build()
		// Enter tallys into classifier
		indx16 = uint16(indx)
		
		if tally.Reset() {
			for eof = false; !eof; {
				tok, score, eof = tally.Next() // get the next one
				scorelog = uint64(math.Log(float64(score) / 1000) * 1000)
				if scorelog > 0 {
					i, _ = rules.Find(tok)
					res[i] = append(res[i], scorer{indx16, scorelog})
				}
			}
		}
	}
	
	t.res = res
	t.rules = *rules
}

// Classify classifies tokens and returns a slice of uint64 where each index is the same as the index for the category name in classifier.Categories, which is the same as the []string of categories originally past to DefineCategories.
func (t *Classifier) Classify(tokens [][]byte) []uint64 {
	var tok []byte
	var ok bool
	var i int
	var obj scorer
	scoreboard := make([]uint64, len(t.Categories))
	for _, tok = range tokens {
		if i, ok = t.rules.Find(tok); ok {
			for _, obj = range t.res[i] {
				scoreboard[obj.category] += obj.score
			}
		}
	}
	return scoreboard
}

// ClassifySimple is a wrapper for Classify, it returns the name of the best category as a string, and the score of the best category as float32.
func (t *Classifier) ClassifySimple(tokens [][]byte) ([]byte, uint64) {
	scoreboard := t.Classify(tokens)
	var bestscore uint64
	var bestcat int
	for cat, score := range scoreboard {
		if score > bestscore {
			bestscore = score
			bestcat = cat
		}
	}
	return t.Categories[bestcat], bestscore
}

func (t *Trainer) classifyTestDoc(test *binsearch.CounterBytes) int {
	var tok []byte
	var v, i int
	var v64 uint64
	var eof, ok bool
	var obj scorer
	scoreboard := make([]uint64, len(t.Categories))
	if test.Reset() {
		for !eof {
			tok, v, eof = test.Next() // get the next one
			if i, ok = t.rules.Find(tok); ok {
				v64 = uint64(v)
				for _, obj = range t.res[i] {
					scoreboard[obj.category] += obj.score * v64
				}
			}
		}
	}
	var bestscore uint64
	i = 0
	for cat, score := range scoreboard {
		if score > bestscore {
			bestscore = score
			i = cat
		}
	}
	return i
}

// Test tries 2,401 different combinations of allowance & maxscore then returns the values of allowance & maxscore which performs the best. Test requires an argument of true or false for verbose, if true Test will print all results to Stdout. 
func (t *Trainer) Test(verbose bool) (float32, float32, error) {
	// Check there are test files
	if t.numTestDocs == 0 {
		return 0, 0, errors.New(`Test: Add test files`)
	}
	num_test_docs := float32(t.numTestDocs)
	// Set some variables
	var bestaccuracy, bestallowance, bestmaxscore, accuracy, allowance, maxscore float32
	var i, indx, correct, l, compare int
	// auto is the list of numbers to try for allowance and maxscore
	var auto_allowance = [...]float32{0,1.05,1.1,1.15,1.2,1.25,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.5,3,4,5,6,7,8,9,10,15,20,25,30,40,50,75,100,150,200,300,400,500,600,700,800,900,1000,1500,2000,3000,4000,5000,10000,20000,50000,100000,1000000}
	var auto_maxscore = [...]float32{0,10000000,1000000,100000,50000,20000,10000,5000,4000,3000,2000,1500,1200,1000,900,800,700,600,550,500,475,450,425,400,375,350,325,300,275,250,225,200,150,100,75,50,40,30,25,20,15,10,8,6,4,2}
	for _, allowance = range auto_allowance { // loop through auto for allowance
		for _, maxscore = range auto_maxscore { // loop through auto for maxscore
			t.Create(allowance, maxscore) // build the classifier for allowance & maxscore
			correct = 0
			// Count the number of correct results from testDocs under this classifier
			for indx = range t.Categories {
				l = len(t.testDocs[indx])
				for i=0; i<l; i++ {
					if compare = t.classifyTestDoc(&t.testDocs[indx][i]); compare == indx {
						correct++
					}
				}
			}
			// And the accuracy is:
			accuracy = float32(correct) / num_test_docs
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
	categories := make([][]byte, numcats)
	for i=0; i<numcats; i++ {
		categories[i] = r.Readx(int(r.Read8()))
	}
	numrules := r.Read64()
	res := make([][]scorer, numrules)
	
	var i2, score uint64
	var id, n uint16
	if numcats < 256 {
		for i2=0; i2<numrules; i2++ {
			n = uint16(r.Read8())
			lst := make([]scorer, n)
			for i=0; i<n; i++ {
				id = uint16(r.Read8())
				score = r.Read64Variable()
				lst[i] = scorer{id, score}
			}
			res[i2] = lst
		}
	} else {
		for i2=0; i2<numrules; i2++ {
			n = r.Read16()
			lst := make([]scorer, n)
			for i=0; i<n; i++ {
				id = r.Read16()
				score = r.Read64Variable()
				lst[i] = scorer{id, score}
			}
			res[i2] = lst
		}
	}
	
	// Create the new object
	t := new(Classifier)
	t.Categories = categories
	t.res = res
	
	// Load binsearch.KeyBytes
	t.rules.Read(r)
	
	// Make sure we're at the end and the checksum is OK
	err = r.EOF()
	if err != nil {
		return nil, errors.New(`Not a valid classifier file.`)
	}
	
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
	
	var lst []scorer
	var obj scorer
	numcats := uint16(len(t.Categories))
	
	w.Write16(numcats)
	for _, cat := range t.Categories {
		w.WriteString8(string(cat))
	}
	w.Write64(uint64(len(t.res)))

	if numcats < 256 {
		for _, lst = range t.res {
			w.Write8(uint8(len(lst)))
			for _, obj = range lst {
				w.Write8(uint8(obj.category))
				w.Write64Variable(obj.score)
			}
		}
	} else {
		for _, lst = range t.res {
			w.Write16(uint16(len(lst)))
			for _, obj = range lst {
				w.Write16(obj.category)
				w.Write64Variable(obj.score)
			}
		}
	}
	
	// Write the binsearch structure
	t.rules.Write(w)
	
	return nil
}
