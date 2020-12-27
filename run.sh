# python plan.py --checkpoint test/checkpoint.txt
# python plan.py --checkpoint collision2/checkpoint.txt --workdir collision2

# python plan.py --input pts.txt --init --workdir route1 --its 1500 --lr 1e-2
# python plan.py --input pts.txt --checkpoint route1/checkpoint.txt --workdir route1 --its 1000 --lr 1e-2
# python plan.py --input pts.txt --checkpoint route1/checkpoint.txt --workdir route1 --its 1000 --lr 1e-3
# python plan.py --input pts.txt --checkpoint route1/checkpoint.txt --workdir route1 --its 1000 --lr 1e-4
# python plan.py --input pts.txt --checkpoint route1/checkpoint.txt --workdir route1 --its 1000 --lr 1e-5

# python plan.py --input pts2.txt --init --workdir route2 --its 1500 --lr 1e-2
# python plan.py --input pts2.txt --checkpoint route2/checkpoint.txt --workdir route2 --its 1000 --lr 1e-2
# python plan.py --input pts2.txt --checkpoint route2/checkpoint.txt --workdir route2 --its 1000 --lr 1e-3
# python plan.py --input pts2.txt --checkpoint route2/checkpoint.txt --workdir route2 --its 1000 --lr 1e-4
# python plan.py --input pts2.txt --checkpoint route2/checkpoint.txt --workdir route2 --its 1000 --lr 1e-5

python plan.py --input pts3.txt --init --workdir route3 --its 1500 --lr 1e-2
python plan.py --input pts3.txt --checkpoint route3/checkpoint.txt --workdir route3 --its 1500 --lr 1e-2
python plan.py --input pts3.txt --checkpoint route3/checkpoint.txt --workdir route3 --its 2000 --lr 1e-3
python plan.py --input pts3.txt --checkpoint route3/checkpoint.txt --workdir route3 --its 2000 --lr 1e-4
python plan.py --input pts3.txt --checkpoint route3/checkpoint.txt --workdir route3 --its 1000 --lr 1e-5