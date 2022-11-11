import string
import nltk

nltk.download('stopwords')
# Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


# defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stopwords])


def preprocessing(description):
    # remove punctuation
    description = remove_punctuation(str(description))

    # lowering the text
    description = description.lower()

    # remove stopwords
    description = remove_stopwords(description)

    return description


def convert_to_list(string_shape_list):
    try:
        return string_shape_list.replace("[", "").replace("]", "").split(",")
    except AttributeError:
        return None
