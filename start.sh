#!/bin/bash 

set -x
source ./pgl_deepwalk.cfg
source ./local_config

unset http_proxy https_proxy

# mkdir workspace
if [ ! -d ${BASE} ]; then
	mkdir ${BASE}
fi


type=$1
ID=$2
server=$3
export PADDLE_PSERVERS_IP_PORT_LIST="10.178.175.107:2222,10.178.162.58:2223"
export PADDLE_PSERVER_PORT_ARRAY=(2222 2223)
export IP_LIST=(10.178.175.107 10.178.162.58)
export PADDLE_PSERVERS_NUM=2
export PADDLE_TRAINERS_NUM=2

if [ "$type" == "kill" ];then
	for i in `ps auxf|grep cluster_train |awk '{print $2}'`; do kill -9 $i; done
        for i in `ps auxf|grep single_train |awk '{print $2}'`; do kill -9 $i; done
elif [ "$type" == "ps" ]; then
    echo "start ps server: ${i}"
    export  TRAINING_ROLE="PSERVER"
    export PADDLE_PORT=${PADDLE_PSERVER_PORT_ARRAY[$ID]}	
    export POD_IP=${IP_LIST[$ID]}

    TRAINING_ROLE="PSERVER" PADDLE_TRAINER_ID=${ID} bash job.sh &> $BASE/pserver.${ID}.log & 
else

    TRAINING_ROLE="TRAINER" PADDLE_TRAINER_ID=${ID} bash job.sh &> $BASE/worker.${ID}.log &
fi
