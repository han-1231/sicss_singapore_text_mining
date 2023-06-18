#install and load necessary packages
#import demo data 
library(readr)
df_twitter <- read_csv("/Users/han/Desktop/twitter_data_demo.csv")#replace the path with your own path
#extract the text column and save as a new dataframe
df_text <- df_twitter[,3]
# Install
install.packages("tm")  # for text mining
install.packages(c("tidyverse","tidytext")) # for text mining
install.packages("SnowballC") # for text stemming
install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer") # color palettes
install.packages("syuzhet") # for sentiment analysis
install.packages("ggplot2") # for plotting graphs
# Load
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library("syuzhet")
library("ggplot2")
library("tidyverse")
library("tidytext")
library("wordcloud2")

#-----------------------------------------------#

# Load the data as a corpus
TextDoc <- Corpus(VectorSource(df_text$text))
TextDoc$content[5]
#example: "@mattgaetz wore a gas mask on the House floor to mock coronavirus concerns. Days later, one of his constituents died from it. @GOP #GOP #Florida #coronavirus \nhttps://t.co/9l6LqFnG1t"
#-----------------------------------------------#
#data preprocessing: remove URLs, mentioned username, punctuations and emojis, newline characters and other symbols, convert text to lower case, remove english common stopwords, eliminate extra white spaces, text stemming

# Use tm_map() to apply content_transformer to your text
toSpace <- content_transformer(function (x,pattern) gsub(pattern, " ", x))
#remove URLs
TextDoc <- tm_map(TextDoc,toSpace,"http[s]?://[^[:space:]]+")
#remove mentioned username
TextDoc <- tm_map(TextDoc,toSpace,"@\\w+")
#remove Newline character
TextDoc <- tm_map(TextDoc,toSpace,"\n")
#remove meaningless symbols
TextDoc <- tm_map(TextDoc,toSpace,"&amp")
TextDoc <- tm_map(TextDoc,toSpace,"rt")

#remove emojis and emoticons
remove_emojis <- function(text) {
  text <- gsub("\\p{So}", "", text, perl = TRUE)
  text <- gsub("\\p{Sk}", "", text, perl = TRUE)
  return(text)
}
TextDoc <- tm_map(TextDoc, remove_emojis)
#remove punctuations and numbers
TextDoc <- tm_map(TextDoc, removePunctuation)
#remove punctuations that are not covered by removePunctuation
TextDoc <- tm_map(TextDoc,toSpace,"[‘’“”•ー—]")
#remove numbers
TextDoc <- tm_map(TextDoc, removeNumbers)
# Convert the text to lower case
TextDoc <- tm_map(TextDoc, content_transformer(tolower))
# Remove english common stopwords
TextDoc <- tm_map(TextDoc, removeWords, stopwords("english"))
#specify your custom stopwords
TextDoc <- tm_map(TextDoc, removeWords, c("can")) 
# Text stemming - which reduces words to their root form, e.g. from "died" to "die"
TextDoc <- tm_map(TextDoc, stemDocument)
# Eliminate extra white spaces
TextDoc <- tm_map(TextDoc, stripWhitespace)

# Extract the preprocessed text from the corpus
preprocessed_texts <- sapply(TextDoc, as.character)
df_text$clean_text <- preprocessed_texts

#-----------------------------------------------#
#building the term document matrix

# Build a term-document matrix
TextDoc_tdm <- TermDocumentMatrix(TextDoc)
tdm_m <- as.matrix(TextDoc_tdm)
# Sort by descearing value of frequency
tdm_v <- sort(rowSums(tdm_m),decreasing=TRUE)
tdm_d <- data.frame(word = names(tdm_v),freq=tdm_v)
# Display the top 5 most frequent words
head(tdm_d, 5)

# Plot the most frequent words
barplot(tdm_d[1:5,]$freq, las = 2, names.arg = tdm_d[1:5,]$word,
        col ="yellow", main ="Top 5 most frequent words",
        ylab = "Word frequencies")

#generate word cloud
set.seed(1234)
wordcloud(words = tdm_d$word, freq = tdm_d$freq, min.freq = 20,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"),scale = c(3.5,0.25))

wordcloud2(data=tdm_d, size=2, color='random-dark')


# Find associations 
findAssocs(TextDoc_tdm, terms = c("coronavirus"), corlimit = 0.05)	
findAssocs(TextDoc_tdm, terms = c("trump"), corlimit = 0.15)	


#applying TF-IDF computation
#convert tdm to a tidy format
tidy_tdm <- tidy(TextDoc_tdm)

# Calculate tf-idf
tidy_tdm <- tidy_tdm %>%
  bind_tf_idf(term, document, count)

# Select top 10 terms with highest TF-IDF in each document
top_terms <- tidy_tdm %>%
  group_by(document) %>%
  top_n(5, tf_idf)

top_terms <- subset(top_terms,as.numeric(document)<=5)

# Plot
ggplot(top_terms, aes(x = reorder(term, tf_idf), y = tf_idf, fill = document)) +geom_col(show.legend = FALSE) +facet_wrap(~document, scales = "free") + coord_flip() +labs(x = "Terms",y = "tf-idf",title = "Top terms in each document by tf-idf")

#counting and filtering biagrams
#filtering all bigrams from each documents

bigram <- df_text%>%unnest_tokens(bigram,clean_text,token = "ngrams",n=2,drop = FALSE)

#count the bigrams 
bigram_top <- bigram%>%count(bigram,sort = TRUE)
bigram_separate <- bigram%>%
  separate(bigram,c("word1","word2"),sep = " ")
bigram_unite <- bigram_separate %>%
  unite(bigram,word1,word2,sep=" ")
#explore bigrams
covid_word <- bigram_separate %>%
  filter(word1 == "coronavirus") %>%
  count(text, word2, sort = TRUE)
#-----------------------------------------------#
#sentiment and emotion analysis
# regular sentiment score using get_sentiment() function and method of your choice
# please note that different methods may have different scales

syuzhet_vector <- get_sentiment(df_text$clean_text, method="syuzhet")
# see the first row of the vector
head(syuzhet_vector)
# see summary statistics of the vector
summary(syuzhet_vector)

# bing
bing_vector <- get_sentiment(df_text$clean_text, method="bing")
head(bing_vector)
summary(bing_vector)
#affin
afinn_vector <- get_sentiment(df_text$clean_text, method="afinn")
head(afinn_vector)
summary(afinn_vector)
#compare the first row of each vector using sign function
rbind(
  sign(head(syuzhet_vector)),
  sign(head(bing_vector)),
  sign(head(afinn_vector))
)

#emotional classification using NRC Word-Emotion Associatioin Lexicon
# run nrc sentiment analysis to return data frame with each row classified as one of the following
# emotions, rather than a score: 
# anger, anticipation, disgust, fear, joy, sadness, surprise, trust 
# It also counts the number of positive and negative emotions found in each row
df_emotion<-get_nrc_sentiment(df_text$clean_text[1:100])
# head(d,10) - to see top 10 lines of the get_nrc_sentiment dataframe
head (df_emotion,10)

#visualization
#transpose
td<-data.frame(t(df_emotion))
#The function rowSums computes column sums across rows for each level of a grouping variable.
td_new <- data.frame(rowSums(td))
#Transformation and cleaning
names(td_new)[1] <- "count"
td_new <- cbind("sentiment" = rownames(td_new), td_new)
rownames(td_new) <- NULL
#Plot One - count of words associated with each sentiment
quickplot(sentiment, data=td_new, weight=count, geom="bar", fill=sentiment, ylab="count")+ggtitle("Tweet emotions")

#Plot two - count of words associated with each sentiment, expressed as a percentage
barplot(
  sort(colSums(prop.table(df_emotion))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  las = 1, 
  main = "Emotions in Text", xlab="Percentage")

