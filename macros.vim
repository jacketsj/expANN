nmap <leader>f :term ++shell ./build.sh && build/expann<CR>
nmap <leader>d :term ++shell ./build.sh && gdbgui build/expann<CR>
nmap <leader>d :term ++shell ./build.sh && gdbgui build/expann<CR>
nmap <leader>p :term ++shell python3 src/pyplotter.py<CR>
nmap <leader>c :term ++shell ./build.sh<CR>
nmap <leader>g :term ++shell ./build.sh && gdb -ex run build/expann<CR>
nmap <leader>v :term ++shell ./build.sh && valgrind build/expann<CR>
"nmap <leader>u :term ++shell ./build.sh && perf record -a build/expann<CR>
nmap <leader>u :term ++shell ./build.sh --stack-info && perf record -F 60000 -a -g build/expann && hotspot<CR>
nmap <leader>i :term ++shell ./build.sh --stack-info && valgrind --tool=callgrind build/expann && kcachegrind<CR>
nmap <leader>I :term ++shell ./build.sh --stack-info && /snap/bin/valgrind --tool=callgrind --collect-atstart=no --instr-atstart=no build/expann && kcachegrind<CR>
nmap <leader>o :term ++shell ./build.sh --gcov && build/expann && gcovr --html-details gcov_report/report.html && xdg-open gcov_report/report.html<CR>
