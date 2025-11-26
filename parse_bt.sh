# usage: <bin/so path> <symbol_file>
while IFS= read -r line; do
  #echo "$line"
  addr=`echo "$line" | awk '{print $1}'`
  if [[ $addr != "libc" ]]; then
    #echo $addr
    func_line=`addr2line -f -e $1 $addr`
    func_name=`echo $func_line | awk '{print $1}' | c++filt `
    func_line=`echo $func_line | awk '{print $2}'`
    echo $func_name $func_line
  else
    echo -e "\n"
  fi
done < $2
