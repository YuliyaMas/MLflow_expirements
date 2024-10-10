import string, jieba
from janome.tokenizer import Tokenizer
from nltk.tokenize import WordPunctTokenizer


class WritingSystem:

    def __init__(self, text: str):

        if len(text) > 400:
            lenght = round(len(text) / 10)
            self.text = text[0:lenght]

        else:
            self.text = text

    def cjk_or_not(self):

        kana_token = 0
        nb_token = 0
        cjk_token = 0
        for token in self.text:
            nb_token += 1
            if token in [chr(i) for i in range(0x3040, 0x309F)] or token in [chr(i) for i in range(0x30A0, 0x30FF)]:
                kana_token += 1
            if token in [chr(i) for i in range(0x4E00, 0x9FFF)]:
                cjk_token += 1

        if cjk_token / nb_token * 100 > 90:
            self.language = 'Chinese'
            return self.language

        elif kana_token / nb_token * 100 > 20:
            self.language = 'Japanese'
            return self.language

        else:
            self.language = 'Other'
            return self.language


class Cleaner:

    def __init__(self):

        # asking about writing system

        # setting punctuation lists
        self.punctuation = list(string.punctuation)
        self.more_punct = ["—", "...", "–", "…", ""]
        self.cjk_punct = (
                [chr(i) for i in range(0x3000, 0x3041)] +
                [chr(i) for i in range(0x30fb, 0x3100)] +
                [chr(i) for i in range(0xff1a, 0xff21)] +
                [chr(i) for i in range(0xff3b, 0xff41)] +
                [chr(i) for i in range(0xff5b, 0xff66)]
        )
        self.more_cjk_punct = ["（", "）", "“", "”", "！", "，", "!!"]
        self.punctuation = self.punctuation + self.more_punct + self.cjk_punct + self.more_cjk_punct

        # setting the japanese and chinese tokenizers
        self.jp_tagger = Tokenizer(wakati=True)

        # setting alphabet writing system tokenizer 
        self.nltk = WordPunctTokenizer()

    def tokenizer(self, text: str, punctuation: bool = True):

        self.ws = WritingSystem(text)
        self.new_text = ''

        if self.ws.cjk_or_not() == 'Japanese':
            jp_text = [token for token in self.jp_tagger.tokenize(text)]
            if punctuation == True:
                for token in jp_text:
                    if token not in self.punctuation:
                        self.new_text += token + ' '
                return self.new_text
            else:
                for token in jp_text:
                    self.new_text += token + ' '
                return self.new_text

        elif self.ws.cjk_or_not() == 'Chinese':
            if punctuation == True:
                for token in text:

                    if token not in self.punctuation:
                        self.new_text += token + ''

                cn_text = jieba.cut(self.new_text, cut_all=False)
                self.new_text = ' '.join(cn_text)

                return self.new_text

            elif punctuation == False:

                cn_text = jieba.cut(text, cut_all=False)
                self.new_text = ' '.join(cn_text)
                return self.new_text

        else:
            if punctuation == True:
                for token in text:
                    if token not in self.punctuation:
                        self.new_text += token + ''

                return self.new_text

            else:
                sentence = self.nltk.tokenize(text)
                self.new_text = ' '.join(sentence)
                return self.new_text


if __name__ == '__main__':
    c = Cleaner()
    with open("CORPORA/1_TRAIN/ab_1.txt", "r", encoding="utf-8") as lang_corpus_file:
        corpus_text = lang_corpus_file.read()
        print(corpus_text[0:50])
        preprocessed_text = c.tokenizer(corpus_text)
        with open("CORPORA/1_TRAIN/PREPROCESS/ab.txt", "w", encoding='utf-8') as file:
            file.write(preprocessed_text)
