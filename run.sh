elements=(8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768)

for element in "${elements[@]}"
do
    xmake run singleCoreGEMM $element
done
