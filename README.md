#*currently broken - new deadline 29/1/15*

##description
+ files are cut in blocks of 512 byte
+ unique blocks are stored in a huge dump file and organized in a central index journal 
+ deduplicated file is represented by a text file containing indices of the original blocks 

##dependencies
+ **GNU/Linux**
+ [gcc]
+ [libssl-dev]
+ [cuda-sdk]

##usage
- deduplicate \<filename\>
- reassemble \<metafile\>



[BSD 3 License]: https://tldrlegal.com/license/bsd-3-clause-license-%28revised%29#fulltext
[cuda-sdk]: http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/
[gcc]: https://gcc.gnu.org
[libssl-dev]: https://packages.debian.org/de/wheezy/libssl-dev