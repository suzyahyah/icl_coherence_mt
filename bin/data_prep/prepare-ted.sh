#!/usr/bin/env bash
# Author: Suzanna Sia

DATAD=$(pwd)/data
HOMED=$(pwd)

RUN_STAGE=(0 1 2 3 4)
LANGS=(fr pt de)

for run in ${RUN_STAGE[@]}; do
  if [ $run -eq 0 ]; then
      echo "Downloading data..."

    rm -r $DATAD/TED
    mkdir -p $DATAD/TED
    mkdir -p $DATAD/TED/train
    mkdir -p $DATAD/TED/valid
    mkdir -p $DATAD/TED/test
    mkdir -p $DATAD/TED/multitask_train

    wget https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/multitarget-ted.tgz -O $DATAD/TED/multitarget-ted.tgz
    cd $DATAD/TED

    tar zxvf multitarget-ted.tgz
    rm multitarget-ted.tgz
  fi

for lg in ${LANGS[@]}; do

  if [ $run -eq 1 ]; then
    echo "Combine monolingual files into bitext format.."
    cd $DATAD/TED

    cp multitarget-ted/en-${lg}/raw/ted_train_en-${lg}.raw.en train/en.txt.raw
    cp multitarget-ted/en-${lg}/raw/ted_train_en-${lg}.raw.${lg} train/${lg}.txt.raw

    cp multitarget-ted/en-${lg}/raw/ted_test1_en-${lg}.raw.en test/en.txt.raw
    cp multitarget-ted/en-${lg}/raw/ted_test1_en-${lg}.raw.${lg} test/${lg}.txt.raw

    cp multitarget-ted/en-${lg}/raw/ted_dev_en-${lg}.raw.en valid/en.txt.raw
    cp multitarget-ted/en-${lg}/raw/ted_dev_en-${lg}.raw.${lg} valid/${lg}.txt.raw
    
    cd $HOMED
    python code/process_data/process_ted.py --combine_docs --lang $lg --data_dir $DATAD
    cp $DATAD/TED/train/en-${lg}.tsv $DATAD/TED/train/en-${lg}.tsv.raw
    cp $DATAD/TED/valid/en-${lg}.tsv $DATAD/TED/valid/en-${lg}.tsv.raw
    cp $DATAD/TED/test/en-${lg}.tsv $DATAD/TED/test/en-${lg}.tsv.raw

  fi

  if [ $run -eq 2 ]; then
    cd $HOMED
    echo "Filtering long and bad lines. This may take a while"
    python code/process_data/process_ted.py --filter_lines --lang $lg --data_dir $DATAD
  fi

  if [ $run -eq 3 ]; then
      TRAIND=$DATAD/TED/train
      VALIDD=$DATAD/TED/valid

      sed -n '1,1000p' $TRAIND/en-${lg}.tsv > $DATAD/TED/multitask_train/en-${lg}.tsv
      sed -n '1001,$p' $TRAIND/en-${lg}.tsv > $TRAIND/en-${lg}.tsv.tmp
      mv $TRAIND/en-${lg}.tsv.tmp $TRAIND/en-${lg}.tsv

      # use everything from here

      head -30000 $TRAIND/en-${lg}.tsv | awk -F "\t" '{print $3}' > $TRAIND/${lg}.txt
      head -30000 $TRAIND/en-${lg}.tsv | awk -F "\t" '{print $2}' > $TRAIND/en.txt

      head -1000 $VALIDD/en-${lg}.tsv | awk -F "\t" '{print $3}' > $VALIDD/${lg}.txt
      head -1000 $VALIDD/en-${lg}.tsv | awk -F "\t" '{print $2}' > $VALIDD/en.txt

    echo "multitask train lines: `wc -l $DATAD/TED/multitask_train/en-$lg.tsv`"
    echo "monlingual train lines: `wc -l $DATAD/TED/train/$lg.txt`"
    echo "monolingual valid lines: `wc -l $DATAD/TED/valid/$lg.txt`"
  fi

  if [ $run -eq 4 ]; then
      echo "Getting doc boundaries..."
      python code/process_data/process_ted.py --doc_sep --lang $lg --data_dir $DATAD
  fi
  
done
done

# what about --doc_sep # give it a better name
