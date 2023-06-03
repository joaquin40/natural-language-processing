## -------------------------------------------------------------------------------
knitr::purl("US Economic News Articles.Rmd")


## -------------------------------------------------------------------------------
pacman::p_load(tidyverse, data.table, tidytext, tm)

df <- fread("./data/US-Economic-News.csv")



## -------------------------------------------------------------------------------
df1 <- select(df, headline, text, relevance) |> 
  filter(relevance != "not sure") |> 
  mutate(relevance = as.factor(relevance))


## -------------------------------------------------------------------------------
replacePunctuation <- function(x) { gsub("[[:punct:]]+", " ", x) }
remove_whitespace <- content_transformer(function(x) gsub("\\s+", "", x))

custom_stop <- data.frame(word = c("--", "</br></br>", "br"), lexicon = "Custom")

stop_words <- rbind(custom_stop, stop_words)
remove_custom_words <- function(x) removeWords(x, stop_words$word)

remove_pattern <- function(x) {
  x <- gsub("</br>", "", x)
}


## -------------------------------------------------------------------------------
df2 <- df1 |> 
  unnest_tokens(output = word,input = text) |> 
  anti_join(stop_words, by = "word") |> 
  count(relevance, word, sort =TRUE)  |> 
  bind_tf_idf(term = word,document = relevance,n = n) |> 
  arrange(desc(tf_idf))

df3 <- df2 |> 
  group_by(relevance) |>
  slice_max(tf_idf, n = 5) |> 
  mutate(word = reorder(word, tf_idf))

tf_idf_gg <- df3 |> 
  ggplot(aes(tf_idf, word, fill = relevance)) + 
  geom_col(show.legend = FALSE) + 
  facet_wrap(~relevance, nrow = 1,scales = "free",
             labeller = labeller(relevance = c("no" = "Words not relevant to US economy", "yes" = "Words relevant to US economy"))) + 
  labs(y = NULL, x = "tf-idf")
tf_idf_gg
ggsave("./images/tf_idf.png", tf_idf_gg)


## -------------------------------------------------------------------------------
us_econ_corpus <- VCorpus(VectorSource(df1$text))
inspect(us_econ_corpus[1:2])
lapply(us_econ_corpus[1:2], as.character)



## -------------------------------------------------------------------------------

replacePunctuation <- function(x) { gsub("[[:punct:]]+", " ", x) }
remove_whitespace <- content_transformer(function(x) gsub("\\s+", "", x))
stop_words
custom_stop <- data.frame(word = c("--", "</br></br>"), lexicon = "Custom")
custom_stop
stop_words <- rbind(custom_stop, stop_words)
remove_custom_words <- function(x) removeWords(x, stop_words$word)
remove_pattern <- function(x) {
  x <- gsub("</br>", "", x)
}

clean <- tm_map(us_econ_corpus, remove_custom_words) # remove stop words 
clean <- tm_map(clean , removePunctuation) # remove punctuation
clean <- tm_map(clean, stemDocument)
clean <- tm_map(clean, content_transformer(remove_pattern))
clean <- tm_map(clean, content_transformer(tolower))
#clean <- tm_map(us_econ_corpus, remove_whitespace )

# examine the final clean corpus
lapply(clean[1], as.character)


## -------------------------------------------------------------------------------
clean_dtm <- DocumentTermMatrix(clean,)
clean_dtm


## -------------------------------------------------------------------------------
set.seed(123)
#clean_dtm |> nrow()
index <- sample(1:7991,replace = FALSE,size = 7000)

clean_dtm_train <- clean_dtm[index,]
clean_dtm_test <- clean_dtm[-index,]



## -------------------------------------------------------------------------------
train_labels <- df1[index, ]$relevance
test_labels  <- df1[-index, ]$relevance


## -------------------------------------------------------------------------------
prop.table(table(train_labels))
prop.table(table(test_labels))


## -------------------------------------------------------------------------------
dtm_freq_train <- removeSparseTerms(clean_dtm_train, 0.97)
dtm_freq_test <- removeSparseTerms(clean_dtm_test, 0.97)


## -------------------------------------------------------------------------------
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}


## -------------------------------------------------------------------------------
us_train <- apply(dtm_freq_train, MARGIN = 2, convert_counts)
us_test  <- apply(dtm_freq_test, MARGIN = 2, convert_counts)



## -------------------------------------------------------------------------------
us_test[1:3, 1:3]


## -------------------------------------------------------------------------------
## Step 3: Training a model on the data ----
library(e1071)

us_classifier <- naiveBayes(us_train, train_labels)



## -------------------------------------------------------------------------------
## Step 4: Evaluating model performance ----
test_pred <- predict(us_classifier, us_test)



## -------------------------------------------------------------------------------
library(gmodels)
CrossTable(test_pred, test_labels, laplace = 1,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

caret::confusionMatrix(test_pred, test_labels,  mode = "prec_recall", positive = "yes")


