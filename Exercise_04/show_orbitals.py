from ase.io import read as myread
import nglview as nv
def all_orbitals(molname,pk,nhomo,nlumo,ntotocc,isosurf=0.02,nfirstview=0,nlastview=5000):
    import subprocess 
    #
    # Click "cube creation kit" on the Orbitals aiidalab page ; note the pk
    #
    print ("Show Orbitals Version n. ",4,"ntotocc = ",ntotocc)
    my_pk = str(pk)

    # !rm -Rf ./cube-kit-pk{my_pk}*
    string = './cube-kit-pk'+my_pk+'*'
    subprocess.check_call(['rm -Rf {}'.format(string)],shell=True)
    
    # !cp /home/jovyan/apps/surfaces/tmp/cube-kit-pk{my_pk}.zip .
    string = '/home/jovyan/apps/surfaces/tmp/cube-kit-pk'+my_pk+'.zip'
    subprocess.check_call(['cp {} .'.format(string)],shell=True)

    #!unzip cube-kit-pk{my_pk}.zip
    string = './cube-kit-pk'+my_pk+'.zip'
    subprocess.check_call(['unzip {}'.format(string)],shell=True)
    
    #!cp run_cube_from_wfn_acetylene.sh ./cube-kit-pk{my_pk}
    string1 = 'run_cube_from_wfn_'+molname+'.sh'
    string2 = './cube-kit-pk'+my_pk
    subprocess.check_call(['cp {} {}'.format(string1,string2)],shell=True)
    
    #!cd ./cube-kit-pk{my_pk} ; bash run_cube_from_wfn_acetylene.sh
    subprocess.check_call(['cd {} ; bash {}'.format(string2,string1)],shell=True)
    
    subprocess.check_call(['ls {}/cubes/'.format(string2)],shell=True)
    
    
    dirname = molname + '_cubes'
    subprocess.check_call(['rm -Rf {}'.format(dirname)],shell=True)  
    subprocess.check_call(['mv {}/cubes {}'.format(string2,dirname)],shell=True)
    
    string = './cube-kit-pk'+my_pk+'*'
    subprocess.check_call(['rm -Rf {}'.format(string)],shell=True)
    
    #
    # Create the visualization of the orbitals
    #
    nfile = 1
    nfilew = ntotocc-nhomo+1
    nhomonow = nhomo
    nlumonow = -1
    mydict = {}

    views =[]
    captions = []
    filenames = []

    for i in range (nhomo+nlumo):
        if (nfile <= nhomo):
            nhomonow = nhomonow-1
            midfix = 'HOMO'
            ind = nhomonow
            strind = '-'+str(ind)
        else:
            nlumonow = nlumonow+1
            midfix = 'LUMO'
            ind = nlumonow
            strind = '+'+str(ind)
        if (ind == 0):
            strind = ''
        totstring = "S0_"+str(nfilew)+'_'+midfix+strind+'.cube'
        nfile = nfile+1
        nfilew = nfilew+1
#        myfile = './cube-kit-pk' + str(my_pk) + '/cubes/' + totstring
        myfile = dirname + '/' + totstring
        
        atoms = myread(myfile)
        filenames.append(myfile)
        print ("Filename: ",myfile)
        fileopen = open(myfile)
        file = myfile
        lines = fileopen.readlines()
        a = (lines[1]) 
        ene=(a[2:10])
        views.append(nv.NGLWidget())
        print ("Energy = ",ene)
        captions.append(" E= "+ene+" eV"+"\n"+midfix+strind)
        views[nfile-2].add_component(nv.ASEStructure(atoms))
        c_2 = views[nfile-2].add_component(file)
        c_2.clear()
        c_2.add_surface(color='blue', isolevelType="value", isolevel=-isosurf, opacity=0.5)
        c_3 = views[nfile-2].add_component(file)
        c_3.clear()
        c_3.add_surface(color='red', isolevelType="value", isolevel=isosurf, opacity=0.5)
    #
    # Visualize the orbitals and energy
    #
    
    import ipywidgets as widgets
    myarray = []
    for a in range(nhomo+nlumo):
        myarray.append(views[a])
    caption =[]
    for l in captions:
        caption.append(widgets.HTML(l))
    combined_w2 = []
    for i in range(len(caption)):
        combined_w2.append(widgets.HBox([myarray[i],caption[i]]))
    combined_widgets = widgets.VBox(combined_w2[nfirstview:nlastview])
    return combined_widgets
