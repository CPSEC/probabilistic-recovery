#!/bin/bash

# parameters: strategy
# isolation {0, 1}
# noise

# dd=(1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)
dd=(1.5)
for d in ${dd[@]}; do
	for i in {1..15}; do
		echo "detection delay $d "
		python3 main.py $1 $2 $d $3
	done
done
if [ "$1" == "0" ]; then
	str_name=rtss
elif [ "$1" == "1" ]; then
	str_name=emsoft
elif [ "$1" == "2" ]; then
	str_name=virtual_sensors
elif [ "$1" == "3" ]; then
	str_name=rtss_nonlinear
fi
echo "${str_name}"


# cat emsoft_no_noise.txt | grep -P '(^,(\s+[-]?\d+[.]\d+)+|detection|Number)' | tr '\n' ' ' | sed 's/detection/\ndetection/g' | grep -vP '^detection delay [-]?\d+[.]\d+\s*$' | sed 's/detection delay //;s/ Number of steps:/,/;s/[[:blank:]]*,[[:blank:]]*/, /;s/,//g' | tr -s [:blank:] ' ' | awk 'BEGIN{i=0}{if(sum[$1] == 0){val[i]=$1;i++;};sum[$1]+=$20; count[$1]++; print $0" 0 0 -10 "sqrt(($2*$2)+($3*$3)+(($4+10)*($4+10)))}END{for(j=1; j<i; j++){print val[j]" "(sum[val[j]]/count[val[j]])}}'

