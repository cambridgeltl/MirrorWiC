task=$1
model=$2
cuda=$3


if [ $task == usim ]
then
    python test_usim.py $model  ../eval_data/usim/usim_en.txt token $cuda 9~13 usim

elif [ $task == cosimlex ]
then
    python test_cosimlex.py $model  ../eval_data/cosimlex/data_en.tsv token $cuda 9~13
    echo "==============="
    echo "find output predictions in the evaluation data directory, and upload to codalab https://competitions.codalab.org/competitions/20905 for evaluation."

elif [ $task == wic ]
then
    python test_wic.py $model  ../eval_data/en_wic token $cuda 100 9~13
    echo "==============="
    echo "find output predictions in the evaluation data directory, and upload to codalab https://competitions.codalab.org/competitions/20010 for evaluation."

elif [ $task == wic-tsv ]
then
    echo "============subtask 1 (def):"
    python test_wic-tsv.py $model  ../eval_data/wic-tsv/wic_tsv_def_wic token $cuda 9~13
    echo "==============="
    echo "find output predictions in the evaluation data directory, and upload to codalab https://competitions.codalab.org/competitions/23683 for evaluation."
    echo "==============="
    echo "=============subtask 2 (hyp):"
    python test_wic-tsv.py $model  ../eval_data/wic-tsv/wic_tsv_hyp_wic token $cuda 9~13
    echo "==============="
    echo "find output predictions in the evaluation data directory, and upload to codalab https://competitions.codalab.org/competitions/23683 for evaluation."
    echo "==============="
    echo "=============subtask 3 (both):"
    python test_wic-tsv.py $model  ../eval_data/wic-tsv/wic_tsv_def_hyp_wic token $cuda 9~13
    echo "==============="
    echo "find output predictions in the evaluation data directory, and upload to codalab https://competitions.codalab.org/competitions/23683 for evaluation."
    echo "==============="
elif [ $task == wsd ]
then
    python test_wsd.py $model ../eval_data/wsd/Evaluation_Datasets/ALL/ALL_test_token.csv token $cuda 9~13

elif [ $task == xlwic ]
then
    for lg in chinese_zh korean_ko croatian_hr estonian_et; do
            echo "===========testing" $lg
            python test_wic.py $model  ../eval_data/xlwic_datasets/xlwic_wn/$lg token $cuda 100 9~13
           
    done

elif [ $task == am2ico ]
then
    for lg in zh ka ja ar; do
        echo "==============testing" $lg
        python test_wic.py $model  ../eval_data/AM2iCo/data/$lg token $cuda 300 9~13
    done
else
    echo "task name error"
    echo "please input task name from wic|wic-tsv|usim|cosimlex|wsd|am2ico|xlwic"
fi