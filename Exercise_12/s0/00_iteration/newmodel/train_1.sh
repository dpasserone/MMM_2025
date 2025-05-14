
if [ ! -f tag_0_finished ] ;then
  { if [ ! -f model.ckpt.index ]; then  dp train --mpi-log=master input.json ; else dp train --mpi-log=master input.json --restart model.ckpt; fi }  1>> train.log 2>> train.log
  if test $? -ne 0; then touch tag_failure_0; fi
fi
if [ ! -f tag_1_finished ] ;then
  dp freeze  1>> train.log 2>> train.log
  touch tag_1_finished
fi

if [ ! -f tag_2_finished ] ;then
  dp compress   1>> train.log 2>> train.log
  touch tag_2_finished
fi

