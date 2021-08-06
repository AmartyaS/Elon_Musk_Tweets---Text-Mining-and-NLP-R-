install.packages("tm")
install.packages("readr")
install.packages("RWeka")
install.packages("plotly")
install.packages("syuzhet")
install.packages("tidytext")
install.packages("wordcloud")
library(tm)
library(readr)
library(RWeka)
library(plotly)
library(syuzhet)
library(ggplot2)
require(graphics)
library(tidytext)
library(wordcloud)

file <- read.csv(file.choose())
corp <- Corpus(VectorSource(file$Text))
inspect(corp)

#Cleaning of Corpus
corp_clean <- tm_map(corp,(tolower))
corp_clean <- tm_map(corp_clean,removeNumbers)
corp_clean <- tm_map(corp_clean,removePunctuation)
corp_clean <- tm_map(corp_clean,removeWords,stopwords())
corp_clean <- tm_map(corp_clean,stripWhitespace)
inspect(corp_clean)

#Making of Term Document Frequency
dtm <- TermDocumentMatrix(corp_clean,control = list(minWordLength=c(1,Inf)))
findFreqTerms(dtm,lowfreq = 2)

#Making Barplot
termFreq <- rowSums(as.matrix(dtm))
termFreq <- subset(termFreq,termFreq>=10)
barplot(termFreq,las=2,col=rainbow(20))

#Making of General Wordcloud
m <- as.matrix(dtm)
wordFreq <- sort(rowSums(m),decreasing = T)
wordcloud(words = names(wordFreq),freq =wordFreq,min.freq = 10,random.order = F,
          colors = rainbow(20) )

#Training the system with Positive and Negative Words
pos <- scan(file.choose(),what = "character", comment.char = ";")
neg <- scan(file.choose(),what = "character",comment.char = ";")
tok_del <- " \\t\\r\\n.!?,;\"()"

#Making wordcloud of positive comments
pos.match <- match(names(wordFreq),pos)
pos.match <- !is.na(pos.match)
pos_word <- wordFreq[pos.match]
wordcloud(words = names(pos_word),freq = wordFreq,min.freq = 5,random.order = F,colors = rainbow(20))

#Making Wordcloud of Negative Comments
neg.match <- match(names(wordFreq),neg)
neg.match <- !is.na(neg.match)
neg_word <- wordFreq[neg.match]
wordcloud(words = names(neg_word),freq = wordFreq,min.freq = 10,random.order = F,colors = rainbow(20))

#Bigrams
min_bi <- 2
bitoken <- NGramTokenizer(corp_clean,Weka_control(min=2,max=2,delimiters=tok_del))
two_words <- data.frame(table(bitoken))
sort_two <- two_words[order(two_words$Freq,decreasing = TRUE),]
sort_two$bitoken
wordcloud(sort_two$bitoken,sort_two$Freq,scale = c(2,0.35),min.freq = min_bi,random.order = F,colors = brewer.pal(8,"Dark2"),max.words = 150)

#Trigrams
min_tri <- 3
tritoken <- NGramTokenizer(corp_clean,Weka_control(min=3,max=3,delimiters=tok_del))
three_words <- data.frame(table(tritoken))
sort_three <- three_words[order(three_words$Freq,decreasing = TRUE),]
sort_three$tritoken
wordcloud(sort_three$tritoken,sort_three$Freq,min.freq = min_tri,scale = c(2,0.35),random.order = F,colors = brewer.pal(8,"Dark2"),max.words = 150)

#Emotion Mining or Sentimental Analysis
sv <- get_sentences(file$Text)

nrc <- get_sentiment(sv,method = "nrc")
bing <- get_sentiment(sv,method = "bing")
afinn <- get_sentiment(sv,method="afinn")
syuzhet <- get_sentiment(sv,method = "syuzhet")

sentiment <- data.frame(nrc,bing,afinn,syuzhet)
sentiment

#Most Positive and Negative Comment through NRC Method
pos_nrc <- sv[which.max(nrc)]
neg_nrc <- sv[which.min(nrc)]
pos_nrc
neg_nrc

#Most Positive and Negative Comment through Bing Method
pos_bing <- sv[which.max(bing)]
neg_bing <- sv[which.min(bing)]
pos_bing
neg_bing

#Most Positive and Negative Comment through Afinn Method
pos_afinn <- sv[which.max(afinn)]
neg_afinn <- sv[which.min(afinn)]
pos_afinn
neg_afinn

#Most Positive and Negative Comment through Syuzhet Method
pos_syuzhet <- sv[which.max(syuzhet)]
neg_syuzhet <- sv[which.min(syuzhet)]
pos_syuzhet
neg_syuzhet

#Experimenting with NRC Emotions
emotion <- get_nrc_sentiment(sv)
emo_bar <- colSums(emotion)
barplot(emo_bar,col = rainbow(10),las=1,horiz = T, main = "Emotions",xlab = "Narrative Time")

#Plots with other sentiments

plot(nrc[(1:200)],type="l",main = "NRC Method", xlab = "Narrative Time",
     ylab = "Emotional Valence")
abline(h=0,col="red")


plot(bing[(1:200)],type="l",main = "Bing Method", xlab = "Narrative Time",
     ylab = "Emotional Valence")
abline(h=0,col="red")


plot(afinn[(1:200)],type="l",main = "Afinn Method", xlab = "Narrative Time",
     ylab = "Emotional Valence")
abline(h=0,col="red")


plot(syuzhet[(1:200)],type="l",main = "Syuzhet Method", xlab = "Narrative Time",
     ylab = "Emotional Valence")
abline(h=0,col="red")


#Plots with transformed sentiment values
fit_nrc <- get_transformed_values(nrc,
                                  low_pass_size = 3,
                                  x_reverse_len = 100,
                                  scale_vals=TRUE,
                                  scale_range =F)
plot(fit_nrc, type = "h", main = "Transformed NRC curve",
     xlab = "Narrative Time",
     ylab = "Emotional Valence",col=brewer.pal(8,"Dark2"))



fit_bing <- get_transformed_values(bing,
                                  low_pass_size = 3,
                                  x_reverse_len = 100,
                                  scale_vals=TRUE,
                                  scale_range =F)
plot(fit_bing, type = "h", main = "Transformed Bing curve",
     xlab = "Narrative Time",
     ylab = "Emotional Valence",col=brewer.pal(8,"Dark2"))



fit_afinn <- get_transformed_values(afinn,
                                  low_pass_size = 3,
                                  x_reverse_len = 100,
                                  scale_vals=TRUE,
                                  scale_range =F)
plot(fit_afinn, type = "h", main = "Transformed Afinn curve",
     xlab = "Narrative Time",
     ylab = "Emotional Valence",col=brewer.pal(8,"Dark2"))



fit_syuzhet <- get_transformed_values(syuzhet,
                                  low_pass_size = 3,
                                  x_reverse_len = 100,
                                  scale_vals=TRUE,
                                  scale_range =F)
plot(fit_syuzhet, type = "h", main = "Transformed Syuzhet curve",
     xlab = "Narrative Time",
     ylab = "Emotional Valence",col=brewer.pal(8,"Dark2"))
