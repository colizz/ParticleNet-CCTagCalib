import os
import glob

def find_valid_runlist(regdir, bdt_mode):
    abspath = os.path.abspath(os.getcwd())
    samlist = glob.glob(regdir)
    if len(samlist) == 0:
        raise RuntimeError('Cannot find valid samples')

    inputdirs_list = []
    for sam in glob.glob(regdir):
        if os.listdir(os.path.join(abspath, sam, 'Cards'))[0].startswith('bdt'):
            if bdt_mode == 'auto':
                inputdirs = [
                    os.path.join(abspath, sam, 'Cards', bdt, pt) 
                    for bdt in os.listdir(os.path.join(abspath, sam, 'Cards')) for pt in os.listdir(os.path.join(abspath, sam, 'Cards', bdt))
                ]
            else:
                inputdirs = [
                    os.path.join(abspath, sam, 'Cards', 'bdt'+bdtval, pt) 
                    for bdtval in bdt_mode.split(',') for pt in os.listdir(os.path.join(abspath, sam, 'Cards', 'bdt'+bdtval))
                ]
        elif os.listdir(os.path.join(abspath, sam, 'Cards'))[0].startswith('pt'):
            if bdt_mode == 'auto':
                inputdirs = [
                    os.path.join(abspath, sam, 'Cards', pt, bdt) 
                    for pt in os.listdir(os.path.join(abspath, sam, 'Cards')) for bdt in os.listdir(os.path.join(abspath, sam, 'Cards', pt))
                ]
            elif bdt_mode == 'central':
                inputdirs = []
                for pt in os.listdir(os.path.join(abspath, sam, 'Cards')):
                    bdt_list = os.listdir(os.path.join(abspath, sam, 'Cards', pt))
                    assert len(bdt_list) % 2 == 1
                    inputdirs.append(os.path.join(abspath, sam, 'Cards', pt, sorted(bdt_list)[int((len(bdt_list)-1)/2)]))
            else:
                inputdirs = [
                    os.path.join(abspath, sam, 'Cards', pt, 'bdt'+bdtval) 
                    for pt in os.listdir(os.path.join(abspath, sam, 'Cards')) for bdtval in bdt_mode.split(',')
                ]
        for inputdir in inputdirs:
            assert os.path.exists(inputdir)
            
        inputdirs_list.append(inputdirs)
    
    return sum(inputdirs_list, [])
    