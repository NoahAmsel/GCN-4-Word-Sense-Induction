#!/bin/bash

if [ $# -lt 2 ] ; then
    echo "Usage: sup_eval.sh <System Key> outDir [mapping.key] [evaluation.key]"
    exit -1
fi

keyIn=$1

outDir=$2
outDir=`echo ${outDir} | sed -e 's/\/\\s*$//'`

if [ ! -d ${outDir} ] ; then
    echo "${outDir} doesn't exist!"
    exit -1
fi



if [ $# -gt 2 ] ; then
    trainKey=$3
fi


if [ $# -gt 2 ] ; then
    testKey=$4
fi

if [ ! -f ${trainKey} ] ; then
    echo "${trainKey} doesn't exist."
    exit -1
fi



fileName=$(basename $keyIn .key)
keyAllOut="$fileName.all.suppervised.key"
#keyNounsOut="$fileName.noun.suppervised.key"
#keyVerbsOut="$fileName.verb.suppervised.key"

echo "All words"
./create_supervised_keyfile.pl $keyIn $trainKey > ${outDir}/${keyAllOut}

./scorer2 ${outDir}/${keyAllOut} $testKey
#echo "Nouns"
#./create_supervised_keyfile.pl -p n $keyIn $trainKey > ${outDir}/${keyNounsOut}
#./scorer2 ${outDir}/${keyNounsOut} ../keys/senseinduction_test_nouns.key
#echo "Verbs"
#./create_supervised_keyfile.pl -p v $keyIn $trainKey > ${outDir}/${keyVerbsOut}
#./scorer2 ${outDir}/${keyVerbsOut} ../keys/senseinduction_test_verbs.key
