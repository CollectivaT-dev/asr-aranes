import os
import re
from argparse import ArgumentParser, Namespace
import csv
import os
import re
import string
from unicodedata import normalize
from typing import List
from tqdm import tqdm
import pandas as pd

def run_parser() -> Namespace:
    """Run argument parser"""
    parser = ArgumentParser()
    #parser.add_argument("--labels-path", type=str, required=True, help="Path to labels directory")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data root")
    parser.add_argument("--cv-path", type=str, required=False, help="Path to commonvoice corpus")
    parser.add_argument("--lexicon-path", type=str, action='append', required=False, help="Path to prepared lexicon(s)")
    parser.add_argument("--check-vocab", type=bool, required=False, default=False, help="Check if training and dev sets contain words outside lexicon")
    parser.add_argument("--phonemes-path", type=str, required=False, help="Path to prepared phoneme set")
    parser.add_argument("--subset", type=int, required=False, default=0, help="Subset sets to size")
    return parser.parse_args()


def format_df_cv(df: pd.DataFrame, commonvoice_root: str, sr: int = 16000, subset: int = 0, wordset = set(), update_wordset:bool = False) -> pd.DataFrame:
    """Format CommonVoice train/dev/test dataframe and store in data root"""
    df = df[["path", "sentence", "client_id"]]
    if subset:
        df = df[:subset]

    #kaldi_columns = ['f_id', 's_id', 'sent', 'wav']
    #df_kaldi = pd.DataFrame(columns=kaldi_columns)
    d_list = []

    for i, (path, sent, s_id) in tqdm(df.iterrows(), total=df.shape[0]):
        clean_sent = clean_line(sent)
        #f_id = s_id + '_' + path.replace(".wav", "").replace(".mp3", "")  #Skips spk label 
        f_id = path.replace(".wav", "").replace(".mp3", "")
        wav = "sox {commonvoice_root}/clips/{path} -t wav -r {sr} -c 1 -b 16 - |".format(commonvoice_root=commonvoice_root, path=path, sr=sr)
        
        d = {'f_id':f_id, 's_id': s_id, 'sent': clean_sent, 'wav': wav} 
        d_list.append(d)
        
        #df_kaldi.loc[i] = [f_id, f_id, clean_sent, wav]
        
        if update_wordset:
            wordset.update(clean_sent.split())

    df_kaldi = pd.DataFrame.from_dict(d_list)

    return df_kaldi


def format_df_pp(df: pd.DataFrame, pp_root: str, subset: int = 0, wordset = set(), update_wordset:bool = False) -> pd.DataFrame:
    df = df[["path", "sentence", "speaker_id"]]
    if subset:
        df = df[:subset]

    # kaldi_columns = ['f_id', 's_id', 'sent', 'wav']
    # df_kaldi = pd.DataFrame(columns=kaldi_columns)

    d_list = []
    for i, (path, sent, s_id) in tqdm(df.iterrows(), total=df.shape[0]):
        #s_id_formatted = "s" + "%03d"%s_id #Skips spk label
        #f_id = s_id_formatted + '_' + '_'.join(path.split('_')[1:]).replace('/','_')[:-4] #Skips spk label
        f_id = '_'.join(path.split('_')[1:]).replace('/','_')[:-4]
        wav = "{pp_root}/{path}".format(pp_root=pp_root, path=path)
        
        d = {'f_id':f_id, 's_id': s_id, 'sent': sent, 'wav': wav} 
        d_list.append(d)

        #df_kaldi.loc[i] = [f_id, f_id, sent, wav]

        if update_wordset:
            wordset.update(sent.split())

    df_kaldi = pd.DataFrame.from_dict(d_list)

    return df_kaldi

def df_to_data(df: pd.DataFrame, data_path: str, set_name: str, subset: int = 0) -> None:
    if subset:
        df = df[:subset]

    set_path = "{data_path}/{set_name}".format(data_path=data_path, set_name=set_name)
    if not os.path.exists(set_path):
        os.makedirs(set_path)
    else:
        print("WARNING: Overwriting in directory", set_path)

    wav_scp = open("{set_path}/wav.scp".format(set_path=set_path), "w")
    utt2spk = open("{set_path}/utt2spk".format(set_path=set_path), "w")
    spk2utt = open("{set_path}/spk2utt".format(set_path=set_path), "w")
    text = open("{set_path}/text".format(set_path=set_path), "w")

    for i, (f_id, s_id, sent, wav) in df.sort_values("f_id").iterrows():
        wav_scp.write("{f_id} {wav}\n".format(f_id=f_id, wav=wav))
        utt2spk.write("{f_id} {s_id}\n".format(f_id=f_id, s_id=s_id))  # we wont specify spk id here
        spk2utt.write("{s_id} {f_id}\n".format(f_id=f_id, s_id=s_id))
        text.write("{f_id} {sent}\n".format(f_id=f_id, sent=sent))
    wav_scp.close()
    utt2spk.close()
    spk2utt.close()
    text.close()

    print(set_name, "written to",  data_path, "size:", len(df))

# normalize apostrophes, some we will keep
fix_apos = str.maketrans("`‘’", "'''")
# anything else we convert to space, and will squash multiples later
# this will catch things like hyphens where we don't want to concatenate words
all_but_apos = "".join(i for i in string.punctuation if i != "'")
all_but_apos += "–—“”»«…°¿¡―ὑℤ ःং ঃ ং"
clean_punc = str.maketrans(all_but_apos, (" " * len(all_but_apos)))
# keep only apostrophes between word chars => abbreviations
clean_apos = re.compile(r"(\W)'(\W)|'(\W)|(\W)'|^'|'$")
# keep only dashes between word chars => abbreviations
clean_dash = re.compile(r"(\W)-(\W)|-(\W)|(\W)-|^-|-$")
squash_space = re.compile(r"\s{2,}")
# chars not handled by unicodedata.normalize because not compositions
bad_chars = {'Æ': 'AE', 'Ð': 'D', 'Ø': 'O', 'Þ': 'TH', 'Œ': 'OE',
             'æ': 'ae', 'ð': 'd', 'ø': 'o', 'þ': 'th', 'œ': 'oe',
             'ß': 'ss', 'ƒ': 'f'}
clean_chars = str.maketrans(bad_chars)

def clean_line(textin):
    line = textin
    line = line.translate(fix_apos)
    line = line.translate(clean_punc)
    line = re.sub(clean_apos, r"\1 \2", line)
    line = re.sub(clean_dash, r"\1 \2", line)
    line = re.sub(squash_space, r" ", line)
    line = line.strip(' ')
    line = line.translate(clean_chars)
    line = line.lower()

    return line

def prepare_lexicon(data_path: str, source_lexicon_paths: str, source_phones_path: str) -> None:
    """Prepare data/local/lang directory"""

    #TODO: Check if training data has words that are not in dictionary. 
    # with open("{data_path}/train/text".format(data_path=data_path), "r") as f:
    #     train_data = [" ".join(line.split(" ")[1:]).strip() for line in f.readlines()]
    # words = sorted(set([w for sent in train_data for w in sent.split(" ")]))

    lexicon = []
    for path in source_lexicon_paths:
        lexicon += [line for line in open(path, 'r').readlines()]
    lexicon = list(set(lexicon)) #keep unique
    lexicon.sort() 
    lexicon_words = [l.split()[0] for l in lexicon]
    lexicon = ["!SIL sil\n", "<UNK> spn\n"] + lexicon

    nonsilence_phones = [g+"\n" for g in sorted(set([char[:-1] for char in open(source_phones_path, 'r').readlines()]))]
    optional_silence = ["sil\n"]
    silence_phones = ["sil\n", "spn\n"]
    
    if not os.path.exists("{data_path}/local/lang".format(data_path=data_path)):
        os.makedirs("{data_path}/local/lang".format(data_path=data_path))
    
    open("{data_path}/local/lang/lexicon.txt".format(data_path=data_path), "w").writelines(lexicon)
    open("{data_path}/local/lang/nonsilence_phones.txt".format(data_path=data_path), "w").writelines(nonsilence_phones)
    open("{data_path}/local/lang/optional_silence.txt".format(data_path=data_path), "w").writelines(optional_silence)
    open("{data_path}/local/lang/silence_phones.txt".format(data_path=data_path), "w").writelines(silence_phones)
    open("{data_path}/local/lang/extra_questions.txt".format(data_path=data_path), "w").writelines([])

    return lexicon_words
    
def check_unknown_words(df, words_list):
    return []

def prepare_lexicon_naive(data_path: str) -> None:
    """Prepare data/local/lang directory"""
    with open("{data_path}/train/text".format(data_path=data_path), "r") as f:
        train_data = [" ".join(line.split(" ")[1:]).strip() for line in f.readlines()]
    words = sorted(set([w for sent in train_data for w in sent.split(" ")]))
    
    lexicon = ["!SIL sil\n", "<UNK> spn\n"] + [" ".join([word] + list(word))+"\n" for word in words]
    nonsilence_phones = [g+"\n" for g in sorted(set([char for word in words for char in word]))]
    optional_silence = ["sil\n"]
    silence_phones = ["sil\n", "spn\n"]
    
    if not os.path.exists("{data_path}/local/lang".format(data_path=data_path)):
        os.makedirs("{data_path}/local/lang".format(data_path=data_path))
    
    open("{data_path}/local/lang/lexicon.txt".format(data_path=data_path), "w").writelines(lexicon)
    open("{data_path}/local/lang/nonsilence_phones.txt".format(data_path=data_path), "w").writelines(nonsilence_phones)
    open("{data_path}/local/lang/optional_silence.txt".format(data_path=data_path), "w").writelines(optional_silence)
    open("{data_path}/local/lang/silence_phones.txt".format(data_path=data_path), "w").writelines(silence_phones)
    open("{data_path}/local/lang/extra_questions.txt".format(data_path=data_path), "w").writelines([])
    
    
def main(args: Namespace) -> None:
    all_words = set()

    #prepare_lexicon_naive(args.data_path)
    if args.lexicon_path and args.phonemes_path:
        lexicon_words = prepare_lexicon(args.data_path, args.lexicon_path, args.phonemes_path)
        print("Lexicon with %i entries prepared in"%len(lexicon_words), args.data_path)
    else:
        print("WARNING: Lexicon and phonemes path not given. Not preparing dictionary.")

    if args.cv_path:
        cv_train = pd.read_csv(args.cv_path+"/train.tsv", delimiter="\t")
        cv_dev = pd.read_csv(args.cv_path+"/dev.tsv", delimiter="\t")
        cv_test = pd.read_csv(args.cv_path+"/test.tsv", delimiter="\t")

        print("Formatting cv_train")
        cv_train_kaldi = format_df_cv(cv_train, args.cv_path, subset=args.subset, wordset=all_words, update_wordset=args.check_vocab)
        print("Formatting cv_dev")
        cv_dev_kaldi = format_df_cv(cv_dev, args.cv_path, subset=args.subset, wordset=all_words, update_wordset=args.check_vocab)
        print("Formatting cv_test")
        cv_test_kaldi = format_df_cv(cv_test, args.cv_path, subset=args.subset)

        df_to_data(cv_train_kaldi, args.data_path, "cv_train")
        df_to_data(cv_dev_kaldi, args.data_path, "cv_dev")
        df_to_data(cv_test_kaldi, args.data_path, "cv_test")

        print("Commonvoice data prepared in", args.data_path)
    else:
        print("WARNING: CV path (--cv-path) not given.")


    if args.check_vocab and args.lexicon_path and args.phonemes_path:
        unknown = all_words.difference(lexicon_words)
        unknown_words_path = os.path.join(args.data_path, "unknown_words.txt")
        
        with open(unknown_words_path, 'w') as f:
            for w in unknown:
                f.write(w+"\n")

        print("%i unknown words written to %s"%(len(unknown), unknown_words_path))

    
if __name__ == "__main__":
    args = run_parser()
    main(args)

