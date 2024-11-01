#!/bin/bash
# This script runs sacct for a given job ID

sacct -j $1 --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode

