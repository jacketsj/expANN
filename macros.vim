nmap <leader>f :term ++shell ./build.sh && build/expann<CR>
nmap <leader>d :term ++shell ./build.sh && gdbgui build/expann<CR>
nmap <leader>c :term ++shell ./build.sh<CR>
nmap <leader>g :term ++shell ./build.sh && gdb -ex run build/expann<CR>
nmap <leader>v :term ++shell ./build.sh && valgrind build/expann<CR>
nmap <leader>u :term ++shell ./build.sh && perf record -a build/expann<CR>
