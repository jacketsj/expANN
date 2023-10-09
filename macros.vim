nmap <leader>f :term ++shell ./build.sh && build/expann<CR>
nmap <leader>d :term ++shell ./build.sh && gdbgui build/expann<CR>
nmap <leader>c :term ++shell ./build.sh<CR>
nmap <leader>g :term ++shell ./build.sh && gdb -ex run build/expann<CR>
nmap <leader>v :term ++shell ./build.sh && valgrind build/expann<CR>
"nmap <leader>u :term ++shell ./build.sh && perf record -a build/expann<CR>
nmap <leader>u :term ++shell ./build.sh && perf record -a -g build/expann<CR>
nmap <leader>i :term ++shell ./build.sh && valgrind --tool=callgrind build/expann && kcachegrind<CR>
nmap <leader>I :term ++shell ./build.sh && /snap/bin/valgrind --tool=callgrind --collect-atstart=no --instr-atstart=no build/expann && kcachegrind<CR>
nmap <leader>o :term ++shell covr --html-details gcov_report/report.html && xdg-open gcov_report/report.html<CR>
"TODO edit above to also build the gcov version (after making edits to cmakelists and build.sh)
