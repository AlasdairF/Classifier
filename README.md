##Classifier

This is a very fast and very memory efficient text classifier for [Go](http://golang.org/). It can train and classify thousands of documents in seconds. The resulting classifier can be saved and loaded from file very quickly, using its own custom file format designed for high speed applications. The classifier itself uses my [BinSearch](http://github.com/AlasdairF/BinSearch) package as its structural backend, which is faster than a hashtable while using only 8 - 16 bytes of memory per token, with 5KB overhead (every word in the English language could be included in the classifier and the entire classifier would fit into 7MB of memory.)

This classifier was written after much experience of trying many different classification techniques for the problem of document categorization, and this is my own implementation of what I have found works best. It uses an ensemble method to increase accuracy, which is similar to what is more commonly known as a 'Random Forest' classifier. This classifier is made specifically for document classification; it classifies based on token frequency and rarity whereby if category_1 has 0.01 frequency for a particular token, and the overall average frequency is 0.005 then this token will be given a score of Log(0.01 / 0.005) = 0.693 for this category. Twenty different ensembles of each category are generated, pruned and then combined. Additionally, this classifier is adaptive in that it can self-optimize through the `Test` function. I attempted many other techniques that did not make it into the final version, because they were unsuccessful; this classifier is based on experience and practice, not only theory - it is accurate, fast, efficient and made for production use in high-bandwidth applications.

For people who are not familiar with classifiers: you start with your list of categories and several (more is better) "training files" for each category, which have been hand picked to be good representatives of this category. You then load these categories and training files into the classifier and it magically makes a classifier object which can then be used to classify new documents into these categories.

Due to the use of [BinSearch](http://github.com/AlasdairF/BinSearch) as the backend, the maximum size of any individual category or token length is 64 bytes inclusive, i.e. `[65]byte`.


## Importing

    import "github.com/AlasdairF/Classifier"
	
## Training

Start the trainer:

    obj := new(classifier.Trainer)
	
Define your categories, this must be a slice of a slice of bytes: `[][]byte`.

	obj.DefineCategories(categories)
	
Add training documents, category is a slice of bytes `[]byte`, tokens is a slice of a slice of bytes `[][]byte` (or a slice of uint64 as imported).

	err := obj.AddTrainingDoc(category, tokens)
	// and again for each training document
	
If you are going to use the `Test` feature to optimize for the best variables for training then you need to add test files. If you don't have any test files then you can add the training files as test files too, this will give you a higher than true accuracy report but it will still help the `Test` function determine the best variables for the classifier.

	err := obj.AddTestDoc(category, tokens)
	// keep doing it for each one
	
The classifier uses two variables called `allowance` and `maxscore` to optimize the classifier. Both are `float32`. `allowance` means that any word with a score below this will not be included in the classifier. `maxscore` means that no word can be given a score of more than this in the classifier. It is best to let the `Test` function choose these for you.

To use the `Test` function (once you've added training and test files) you only need to do as follows. Note that if `verbose` is set to true then you will get thousands of lines output to Stdout telling you the accuracy level of each test and which one was best; if it's set to false then it's silent. `Test` returns the best values for `allowance` and `maxscore`.

	verbose := true
	allowance, maxscore, err := obj.Test(verbose)

You can now create your classifier:

	obj.Create(allowance, maxscore)
	
Then save it to a file:

    err := obj.Save(`somedir/myshiz.classifier`)
	
You can also use any of the Classification functions below on your `Trainer` object, if you want to start classifying right away. You only need to create a new `Classifier` object if you are loading a classifier from a file since the `Trainer` object inherits all of the functions of the `Classifier` object.

## Classification

Load the classifier you previously saved:

    obj, err := classifier.Load(`somedir/myshiz.classifier`)
    // *OR*
    obj := classifier.MustLoad(`somedir/myshiz.classifier`)
	
If want to retrieve a list of the categories for printing then they are here as a slice of a slice bytes:

    categories := obj.Categories // [][]byte

Classify something:

    scores := obj.Classify(tokens) // tokens is [][]byte
	
The above will give you a slice of `uint64` where each index represents the index of the category in `obj.Categories` (which is exactly the same as what you gave originally to `DefineCategories`) and the `uint64` is the score for this category (only meaningful relative to the other scores.) You may need to sort this list, for which I would recommend my optimized sorting function [Sort/Uint16Uint64](http://github.com/AlasdairF/Sort) like this:

     // import "github.com/AlasdairF/Sort/Uint16Uint64"
     sorted := sortUint16Uint64.New(scores)
     sortUint16Uint64.Desc(sorted)
     // You could then output this as follows
     cats := obj.Categories
     for i, score := range sorted {
     	fmt.Println(i, `Category`, string(cats[score.K]), `Score`, score.V)
     }

To make things easy, if you want *only* the best matching category and score, and not the results for each category, then you can do the following, which returns the `[]byte` of the category that this document best matches and its score as `uint64`:

    category, score := classifier.ClassifySimple(tokens)
    fmt.Println(`Best category was`, string(category), `with score`, score)
	

## Tokenization / Feature Extraction

You do need to tokenize each document before training on it or classifying it, which means to extract `tokens` (usually words) from the document ready for classifying. How you tokenize depends on what you are trying to classify. However you choose to tokenize, you must be sure to do it *exactly the same* to the training documents, test documents, and the documents you eventually classify.

I have written a [Tokenize](http://github.com/AlasdairF/Tokenize) package that works perfectly with this Classifier. I suggest you check that out.

Following are some recommendations if you choose to do your own tokenization.

1. Make the text all lowercase.
2. If the text is generated by OCR then remove accents from characters to normalize mistakes.
3. Strip punctuation, numbers and special characters.
4. If you know the language then you can use a stemmer to reduce the words to their stems, [like this one I collected](http://github.com/AlasdairF/Stemmer).
5. Remove stopwords, which are common words such as 'and', 'or', 'then', etc.

If you have additional features for the document, such as title, author, keywords, etc. then these can be added to the tokens. You may want to make them special by adding prefix (which could be a capital letter if you lowercased everything) so that they only match against other instances of the same feature (e.g. prefix 'T' onto the beginning of each title token). You may want to add these as normal tokens *and* add them in again with the prefix (which works well for titles and keywords).

Tokens don't have to be split on individual words, you can split them on anything you want, such as bigrams (double words, e.g. 'ancient history'). Or add both single words and then add bigrams as well. Often though bigrams do not work as well as one might expect, since they can easily become too powerful as a scorer and then throw a document into the wrong category if it happens to contain this bigram, not to mention you can turn thousands of tokens into millions by doing this. Long story sort: bigrams can be tricky, they can increase your accuracy but only if you test properly with them, select them well, and ensure they are suitable in your case.


~ Alasdair Forsythe
