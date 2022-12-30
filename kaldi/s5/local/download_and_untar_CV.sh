#!/usr/bin/env bash

# Copyright   2014  Johns Hopkins University (author: Daniel Povey)
#             2017  Luminar Technologies, Inc. (author: Daniel Galvez)
#             2017  Ewald Enzinger
#	            2022  Col·lectivaT, SCCL
# Apache 2.0

# Adapted from egs/mini_librispeech/s5/local/download_and_untar.sh (commit 1cd6d2ac3a935009fdc4184cb8a72ddad98fe7d9)

remove_archive=false

if [ "$1" == --remove-archive ]; then
  remove_archive=true
  shift
fi

if [ $# -ne 2 ]; then
  echo "Usage: $0 [--remove-archive] <data-base> <lang>"
  echo "e.g.: $0 /export/data/ https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz"
  echo "With --remove-archive it will remove the archive after successfully un-tarring it."
fi

data=$1
lang=$2

#TODO: Change this link with newer versions of CV dataset
url=https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-12.0-2022-12-07/cv-corpus-12.0-2022-12-07-oc.tar.gz

echo datadir: $data
echo url: $url

if [ ! -d "$data" ]; then
  echo "$0: no such directory $data"
  exit 1;
fi

if [ -z "$url" ]; then
  echo "$0: empty URL."
  exit 1;
fi

if [ -f $data/$lang/.complete ]; then
  echo "$0: data was already successfully extracted, nothing to do."
  exit 0;
fi

filename=$(basename $url)
filepath="$data/$filename"

if [ -f $filepath ]; then
  size=$(/bin/ls -l $filepath | awk '{print $5}')
  echo TAR size: $size
fi

if [ ! -f $filepath ]; then
  if ! which wget >/dev/null; then
    echo "$0: wget is not installed."
    exit 1;
  fi
  echo "$0: downloading data from $url.  This may take some time, please be patient."

  echo wget --no-check-certificate $url
  cd $data
  if ! wget --no-check-certificate $url; then
    echo "$0: error executing wget $url"
    exit 1;
  fi
fi

cd $data

echo tar -xzf $filepath
if ! tar -xzf $filepath; then
  echo "$0: error un-tarring archive $filepath"
  exit 1;
fi

untardir=$(basename $filepath -$lang.tar.gz)
mv $untardir/$lang .
rm -r $untardir

touch $data/$lang/.complete

echo "$0: Successfully downloaded and un-tarred $filepath"

if $remove_archive; then
  echo "$0: removing $filepath file since --remove-archive option was supplied."
  rm $filepath
fi
