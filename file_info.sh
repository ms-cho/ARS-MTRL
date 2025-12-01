#!/bin/bash

TARGET_DIR="."

function file_info {
  for file in "$1"/*; do
    if [ -d "$file" ]; then
      # 디렉토리일 경우 재귀적으로 함수 호출
      file_info "$file"
    elif [ -f "$file" ]; then
      # 파일일 경우 파일명, 라인 수, 크기 출력
      lines=$(wc -l < "$file" | sed 's/^[[:space:]]*//')
      size=$(wc -c < "$file" | sed 's/^[[:space:]]*//')
      echo "$file : $lines line, $size byte"
    fi
  done
}

file_info "$TARGET_DIR"
