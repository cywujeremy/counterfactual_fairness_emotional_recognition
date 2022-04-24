for session in `ls "E:\Download\IEMOCAP_full_release_withoutVideos\converted_dataset\converted_dataset\IEMOCAP_full_release_converted"`
    do 
        wav_path="E:\\Download\\IEMOCAP_full_release_withoutVideos\\converted_dataset\\converted_dataset\\IEMOCAP_full_release_converted\\"${session}"\\sentences\\wav\\"
        for folder in `ls ${wav_path}`
            do
                for wav_file in `ls $wav_path$folder`
                    do
                        path=$wav_path$folder\\$wav_file
                        new_path=${path%%.wav}\_int.wav
                        sox $path -b 16 -e signed-integer $new_path
                        echo $path converted! 
                    done
            done
    done