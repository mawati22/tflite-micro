#!/bin/bash -e
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Tests an Xtensa binary by parsing the log output.
#
# First argument is the binary location.
#
# Second argument is a regular expression that's required to be in the output
# logs for the test to pass.
#
# Third argument is the TARGET name, here this is xtensa
#

declare -r TEST_TMPDIR=/tmp/test_xtensa_binary/
declare -r MICRO_LOG_PATH=`dirname ${TEST_TMPDIR}/$1`
declare -r MICRO_LOG_FILENAME=${TEST_TMPDIR}/${1}_logs.txt
mkdir -p ${MICRO_LOG_PATH}

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

xt-run --turbo $1 >${MICRO_LOG_FILENAME} 2>&1

if [[ ${2} != "non_test_binary" ]]
then
  if grep -q "$2" ${MICRO_LOG_FILENAME}
  then
    #cat ${MICRO_LOG_FILENAME}
    echo -e "$3, $1 : ${GREEN}PASS${NC}"
    exit 0
  else
    #cat ${MICRO_LOG_FILENAME}
    echo -e "$3, $1 : ${RED}FAIL${NC}. See log ${MICRO_LOG_FILENAME} for details."
    exit 1
  fi
fi