#!/bin/bash

db2 "CONNECT TO TPCDS"
outputdir="new_export"

# One time - export the rows from the explain tables from all successful queries - BEGIN
# export explain tables
# Declare an array of strings
declare -a export_tables=("EXPLAIN_ACTUALS" "EXPLAIN_ARGUMENT" "EXPLAIN_DIAGNOSTIC" "EXPLAIN_DIAGNOSTIC_DATA" "EXPLAIN_INSTANCE" "EXPLAIN_OBJECT" "EXPLAIN_OPERATOR" "EXPLAIN_PREDICATE" "EXPLAIN_STATEMENT" 
"EXPLAIN_STREAM" "ACTIVITYMETRICS_DB2ACTIVITIES" "ACTIVITYSTMT_DB2ACTIVITIES" "ACTIVITYVALS_DB2ACTIVITIES" "ACTIVITY_DB2ACTIVITIES")

for table_name in "${export_tables[@]}"; do
    echo "$table_name"
    # get column names
    db2 -x "describe table $table_name" | awk '{print $1}' | paste -sd, > headers.csv
    # get data
    db2 -x "export to data.csv of del modified by nochardel select * from $table_name"

    # # concatenate column names and data
    cat headers.csv data.csv > "${outputdir}/${table_name}.csv"

    # # remove headers and data files
    rm headers.csv data.csv
done

# Print all elements of the array
echo ${explain_tables[@]}

# Print the first element of the array
echo ${explain_tables[0]}

# Print the second element of the array
echo ${explain_tables[1]}

db2 "CONNECT RESET"