nmap <leader>f :term ++shell ./build.sh && build/expann<CR>
nmap <leader>F :term ++shell ./build.sh && python3 ~/proj/bashcord/bashcord.py "build/expann" --config bashcord_config.json --files plots/newyearnewdata_sift_k_10/*.png<CR>
nmap <leader>d :term ++shell ./build.sh && gdbgui build/expann<CR>
nmap <leader>D :term ++shell ./build.sh && rr record build/expann; gdbgui --gdb-cmd "rr replay --"<CR>
nmap <leader>p :term ++shell python3 src/pyplotter.py<CR>
nmap <leader>c :term ++shell ./build.sh<CR>
nmap <leader>C :term ++shell ./build.sh --clang<CR>
nmap <leader>g :term ++shell ./build.sh && gdb -ex run build/expann<CR>
nmap <leader>v :term ++shell ./build.sh --stack-info && valgrind --track-origins=yes build/expann<CR>
nmap <leader>a :term ++shell ./build.sh --clang --asan && build/expann<CR>
"nmap <leader>u :term ++shell ./build.sh && perf record -a build/expann<CR>
nmap <leader>u :term ++shell ./build.sh --stack-info && perf record -F 60000 -g build/expann && hotspot<CR>
nmap <leader>i :term ++shell ./build.sh --stack-info && valgrind --tool=callgrind build/expann && kcachegrind<CR>
nmap <leader>y :term ++shell ./build.sh --stack-info && valgrind --tool=cachegrind build/expann && kcachegrind<CR>
nmap <leader>I :term ++shell ./build.sh --stack-info && /snap/bin/valgrind --tool=callgrind --collect-atstart=no --instr-atstart=no build/expann && kcachegrind<CR>
nmap <leader>o :term ++shell ./build.sh --gcov && build/expann && gcovr --html-details gcov_report/report.html && xdg-open gcov_report/report.html<CR>
