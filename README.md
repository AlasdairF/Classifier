##Classifier

This is a very fast and very memory efficient text classifier for [Go](http://golang.org/). It can train and classify thousands of documents in seconds. Why? Because most people who write classifiers have no concept of efficiency.

I based this classifier on my experience of trying many different classification techniques for the problem of document categorization, and this is my own implementation of what I have found works best. It uses an ensemble method to increase accuracy, which is similar to a Random Forest. This classifier is made specifically for document classification; it classifies based on token frequency and rarity whereby if category_1 has 0.01 frequency for a particular token, and the overall average frequency is 0.005 then this token will be given a score of (0.01 / 0.005) = 2 for this category. Twenty different ensembles of each category are generated, pruned and then combined. Additionally, this classifier is adaptive in that it can self-optimize through the `Test` function. I did try to implement more fancy stuff and make it re-train itself for higher accuracy, but it only ended up getting over-trained, so those features were removed. What you see here is the best I could do after a lot of trial and error.

For people who are not familiar with classifiers: you start with your list of categories and several (more is better) "training files" for each category, which have been hand picked to be good representatives of this category. You then load these categories and training files into the classifier and it magically makes a classifier object which can then be used to classify new documents into these categories.


## Installation

    go get github.com/AlasdairF/Classifier

	
## Training

Start the trainer:

    classifier := new(Trainer)
	
Define your categories, this must be a slice of strings: `[]string`.

	classifier.DefineCategories(categories)
	
Add training documents, category is a string, tokens is a slice of strings.

	err := classifier.AddTrainingDoc(category,tokens)
	// and again for each training file
	
If you are going to use the `Test` feature to optimize for the best variables for training then you need to add test files. If you don't have any test files then you can add the training files as test files too, this will give you a higher than true accuracy report but it will still help the `Test` function determine the best variables for the classifier.

	err := classifier.AddTestDoc(category,tokens)
	// keep doing it for each one
	
The classifier uses two variables called `allowance` and `maxscore` to optimize the classifier. Both are `float64`. `allowance` means that any word with a score below this will not be included in the classifier. `maxscore` means that no word can be given a score of more than this in the classifier. It is best to let the `Test` function choose these for you.

To use the `Test` function (once you've added training and test files) you only need to do as follows. Note that if `verbose` is set to true then you will get thousands of lines output to Stdout telling you the accuracy level of each test and which one was best; if it's set to false then it's silent.

	verbose := true
    allowance, maxscore, err := classifier.Test(verbose)

You can now create your classifier:

	classifier.Create(allowance,maxscore)
	
Then save it to a file:

    classifier.Save(`myshiz.classifier`)
	
You can also use any of the Classification functions below on your `Trainer` object, if you want to start classifying right away. You only need to create a new object if you are loading a classifier from a file.


## Classification

Start the classifier:

    classifier := new(Classifier)

Load the classifier you previously saved:

    classifier.Load(`myshiz.classifier`)
	
If you need to reload the categories they are here as a slice of strings:

    categories := classifier.Categories // []string

Classify something:

    scores := classifier.Classify(tokens)
	
The above will give you a slice of `float64` where each index represents the index of the category in `classifier.Categories` (which is exactly the same as what you gave originally to `DefineCategories`) and the `float64` is the score for this category. You may need to sort this list.

To make things easy, if you want *only* the best matching category and not the scores then you can do this, which returns only a `string` of the category this document best matches:

    category := classifier.ClassifySimple(tokens)
	

## Tokenization

You do need to tokenize each document before training on it or classifying it, which means make it into a slice of strings (usually words) with each token standardized in some way. How you tokenize depends on what you are trying to classify. However you choose to tokenize, you must be sure to do it *exactly the same* to the training documents, test documents, and the documents you eventually classify.

Following are some recommendations for tokenizing. You will find many useful functions for tokenizing in my [Strings](http://github.com/AlasdairF/Strings) package.

1. Make the text all lowercase.
2. If the text is generated by OCR then remove accents from characters to normalize mistakes.
3. Strip punctuation, number and special characters.
4. If you know the language then you can use a stemmer to reduce the words to their stems, [like this one I collected](http://github.com/AlasdairF/Stemmer).
5. Remove stopwords, which are common words such as 'and', 'or', 'then', etc.

If you have additional features for the document, such as title, author, keywords, etc. then these can be added to the tokens. You may want to make them special by adding a capital letter prefix onto them so that they only match against other instances of the same feature (e.g. prefix 'T' onto the beginning of each title token). You may want to add these in as normal tokens *and* add them in again with the prefix (which works well for titles and keywords).

Tokens don't have to be split on individual words, you can split them on anything you want, such as bigrams (double words, e.g. 'ancient history'). Or add both single words and then add bigrams as well. Often though bigrams do not work as well as one might expect, since they can easily become too powerful as a scorer and then throw a document into the wrong category if it happens to contain this bigram, not to mention you can turn thousands of tokens into millions by doing this. Long story sort: bigrams can be tricky, they can increase your accuracy but only if you test properly with them, select them well, and ensure they are suitable in your case.


~ Alasdair Forsythe
