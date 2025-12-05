#!/bin/bash
set -e

echo "Running Section 3.1.1 ..."
cd section3_1_1 && bash run_codes.sh && cd ..

echo "Running Section 3.1.2 ..."
cd section3_1_2 && bash run_codes.sh && cd ..

echo "Running Section 3.1.3 ..."
cd section3_1_3 && bash run_codes.sh && cd ..

echo "Running Section 3.2 ..."
cd section3_2 && bash run_codes.sh && cd ..

echo "Running Section 3.3 ..."
cd section3_3 && bash run_codes.sh && cd ..

echo "Running Supplementary SI_1 ..."
cd SI_1 && bash run_codes.sh && cd ..

echo "All codes executed successfully."
