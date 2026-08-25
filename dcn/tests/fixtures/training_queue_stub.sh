echo "G1_TEST_QUEUE_STUB_SOURCED" >&2
if [[ "${G1_TEST_REPORT_QUEUE_IN_FLIGHT:-0}" == 1 ]]; then
    echo "G1_TEST_QUEUE_IN_FLIGHT=${TRAINING_QUEUE_IN_FLIGHT-unset}" >&2
fi

enqueue() {
    echo "G1_TEST_QUEUE_STUB_ENQUEUE" >&2
    if [[ "${G1_TEST_REPORT_DATA_GROUP:-0}" == 1 ]]; then
        echo "G1_TEST_QUEUE_DATA_GROUP=${TRAINING_QUEUE_DATA_GROUP-}" >&2
    fi
    return 97
}

drain() {
    echo "G1_TEST_QUEUE_STUB_DRAIN" >&2
    return 97
}
