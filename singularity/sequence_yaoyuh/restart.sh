
RESUME_ID=$1
ARRAY_IDS=$2
NUM_RESTARTS=$3

./first_job.sh ${RESUME_ID} ${ARRAY_IDS}
echo "First Job Submitted..."

sleep 1

./submit.sh ${NUM_RESTARTS}  ${RESUME_ID} ${ARRAY_IDS}