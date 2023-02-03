#!/bin/bash

for d in */ ; do
	new_name="100$d"
	mv $d $new_name

done
