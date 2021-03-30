# ffmpeg required.
# bash ./resample.sh -d /path/to/source/data
# This script will generate resampled (22050Hz) audio files at /path/to/source/data_resampled.

open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}
run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    printf '%.3d' $? >&3
    )&
}
resample(){
    ffmpeg -loglevel panic -i $1 -ar 22050 $2
}

i=0
d=0
N=16
open_sem $N

while getopts d: flag
do
    case "${flag}" in
        d) datapath=${OPTARG};;
    esac
done

basename=${datapath##*/}
outdir=${datapath/$basename/"${basename}_resampled"}

echo "Target Datapath: $datapath";
echo "Target Outdir: $outdir";

if [[ ! -e $outdir ]]; then
    mkdir $outdir
elif [[ ! -d $outdir ]]; then
    echo "$outdir already exists but is not a directory" 1>&2
fi

# Resample and Save at Outdir
for file_indir in $(find $datapath -name "*.wav"); do
    ((i=i+1))
    filename="${file_indir/".wav"/""}"
    file_outdir="$outdir/${filename##*/}.wav"
    run_with_lock resample $file_indir $file_outdir
    echo "$((i-d))/$i [ PASSED]: $file_outdir"
done