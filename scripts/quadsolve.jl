#!/usr/bin/env -S julia --startup-file=no
Base.set_active_project(dirname(@__FILE__) * "/..")
import QuadSolve
for fname = ARGS
    QuadSolve.main(fname)
end
