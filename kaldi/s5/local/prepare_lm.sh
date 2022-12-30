#!/bin/bash

# Modified from Voxforge
# by Chompakorn Chaksangchaichot
# also by Col·lectivaT

. ./path.sh || exit 1;

echo "=== Building a language model ..."

locdata=data/local
loctmp=$locdata/tmp

echo "--- Preparing a corpus from test and train transcripts ..."

# Language model order
order=3
textcorpus=''

. utils/parse_options.sh

# Prepare a LM from train corpus
mkdir -p $loctmp
# cat data/train/text data/dev/text > $loctmp/utt.txt
# cut -f2- -d' ' < $loctmp/utt.txt | sed -e 's:[ ]\+: :g' > $loctmp/traindev.txt
cut -f2- -d' ' < data/train/text | sed -e 's:[ ]\+: :g' > $loctmp/train.txt
# rm $loctmp/utt.txt

# If given, add extra cleaned text corpus to the mix
if [ ! -z "$textcorpus" ]; then
	echo Adding $textcorpus to text corpus
	cat $textcorpus >> $loctmp/train.txt
fi

#Sort corpus
sort -u $loctmp/train.txt > $loctmp/corpus.txt
rm $loctmp/train.txt

echo "prepare_lm.sh: Text corpus stats ($loctmp/corpus.txt)"
wc $loctmp/corpus.txt 

loc=`which ngram-count`;
if [ -z $loc ]; then
	if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
		sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64 
	else
		sdir=$KALDI_ROOT/tools/srilm/bin/i686
	fi
	if [ -f $sdir/ngram-count ]; then
		echo Using SRILM tools from $sdir
		export PATH=$PATH:$sdir
	else
		echo You appear to not have SRILM tools installed, either on your path,
		echo or installed in $sdir.  See tools/install_srilm.sh for installation
		echo instructions.
		exit 1
	fi
fi

ngram-count -order $order -write-vocab $locdata/vocab-full.txt -wbdiscount \
	-text $loctmp/corpus.txt -lm $locdata/lm.arpa

echo "*** Finished building the LM model!"
