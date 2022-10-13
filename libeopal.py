import numpy as np
from numpy import random
import sys, os, time 

from libensemble.executors.executor import Executor
from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, TASK_FAILED

from distgen import * #Generator
from pmd_beamphysics import ParticleGroup

STAT_NAMES = ['t', 's','numParticles','charge','energy','rms_x', 'rms_y', 'rms_s', \
              'rms_px', 'rms_py', 'rms_ps', 'emit_x', 'emit_y', 'emit_s','mean_x', \
              'mean_y', 'mean_s', 'ref_x', 'ref_y', 'ref_z', 'ref_px', 'ref_py', 'ref_pz', \
              'max_x', 'max_y', 'max_s', 'xpx', 'ypy', 'zpz', 'Dx', 'DDx', 'Dy', 'DDy', \
              'Bx_ref', 'By_ref', 'Bz_ref', 'Ex_ref', 'Ey_ref', 'Ez_ref', 'dE', 'dt', 'partsOutside']

def isInDirectory(filepath, directory):
    """Check if filepath is inside directory"""
    # From https://stackoverflow.com/questions/3812849/how-to-check-whether-a-directory-is-a-sub-directory-of-another-directory
    return os.path.realpath(filepath).startswith(os.path.realpath(directory) + os.sep)

def linkDirectory(path, name=''):
    """Make files available in working directory with recursive symbolic links"""
    # Taken from runOPAL???? Paste link here....
    # Check for recursiveness
    if isInDirectory(os.getcwd(), path):
        print(name + ' directory is subdirectory of working directory! Cannot handle this.. bye!')
        sys.exit()
    # lndir and if fails try cp
    if os.system('lndir '+path) != 0:
        #print("lndir failed (possibly doesn't exist on this system), using cp -rs... \n"),
        if os.listdir(path):
            os.system('cp -rs '+path+'/* .')

def parseHeader(filename):
    """Adapted from pyOPALTools:
    https://gitlab.psi.ch/OPAL/pyOPALTools/-/blob/master/opal/parser/SDDSParser.py
    """
    #import pdb; pdb.set_trace()
    nRows = 0
    nParameters = 0
    with open(filename) as f:
        for line in f:
            if 'SDDS' in line:
                nRows += 1 
            elif '&end' in line:
                nRows +=1
            elif '&description' in line:
                nRows += 3
            elif '&parameter' in line:
                nRows += 4
            elif '&column' in line:
                nRows += 5 
            elif '&data' in line:
                nRows += 3
            elif 'OPAL' in line:
                nRows += 1
            elif 'opal-t' in line:
                nRows += 1
            else:
                pass
    return nRows                


class LibeOpal(object):
    """set up libe runs with opal, process data"""
    def __init__(self,sim_specs):
        try:
            # Inputs
            #self.sim_specs     = sim_specs
            self.basefile_name = sim_specs['user']['basefile_name']
            self.stat_file     = sim_specs['user']['basefile_name']+'.stat'
            self.template_file = sim_specs['user']['input_files_path']+self.basefile_name+'.tmpl'
            self.data_file     = sim_specs['user']['input_files_path']+self.basefile_name+'.data'
            self.distgen_file  = sim_specs['user'].get('distgen_file', None) 
            self.laser_filter  = sim_specs['user'].get('laser_filter',1)
            self.xscale        = sim_specs['user'].get('xscales',1)
            self.obj_scale     = sim_specs['user'].get('objective_scales',1)
            self.time_limit    = sim_specs['user']['sim_kill_minutes']*60
            self.obj_keys      = sim_specs['user']['key_dict'].get('objective_keys', [0]) # change this [0]?
            self.dvar_keys     = sim_specs['user']['key_dict']['dvar_keys']
            self.data_keys     = sim_specs['user']['key_dict']['data_keys'] 
            self.penalty_scale = sim_specs['user']['penalty_scale']
            self.cores         = sim_specs['user'].get('cores', 1) 
            self.max_particles = sim_specs['user']['sim_particles']
            self.zstop         = sim_specs['user']['zstop']
            self.zobj          = sim_specs['user'].get('objective_zlocation', None)
            self.outspecs      = sim_specs['out']
            self.f_def         = sim_specs['user'].get('f_def',None)
            self.weights       = sim_specs['user'].get('weights', None)	
            # Outputs
            self.calc_status  = None
            self.output       = np.zeros(1, dtype=self.outspecs) 
            self.raw_obj_vals = {}
            self.scaled_obj_vals  = np.zeros(len(self.obj_keys)) 
            self.checked_obj_vals = {}
            self.penalty = None
            self.dist = None 
        except Exception as e:
            print(e)
            print('Missing a required user input(s), please check call script.')

    def make_gaussian_dist(self,x):
        """Run distgen to make particle distribution"""
        if len(self.xscale)==1:
            # No scales, use raw vals for everything
            rscale = 1
            tscale = 1

        if 'RADIUS' in self.xscale['name']:
            # Load scales
            rindex  = np.where(self.xscale['name']=='RADIUS') 
            rscale  = self.xscale['scale'][rindex][0]
           
        if 'FWHM' in self.xscale['name']:
            # Load scales
            tindex  = np.where(self.xscale['name']=='FWHM')
            tscale  = self.xscale['scale'][tindex][0]

        # Scale inputs for simulation
        radius  = x[rindex][0]*rscale 
        sigmat  = (x[tindex][0]*tscale)/2.355 # converting FWHM to sigma
        t  = 4*sigmat*np.arange(-1, 1, 0.01)
        pt = 1/(sigmat*(2*np.pi)**0.5)*np.exp(-0.5*((t-0)/sigmat)**2)
        X  = np.zeros((len(t),2))
        X[:,0] = t
        X[:,1] = pt
        with open('tdist_file.dat', 'wb') as f:
            np.savetxt(f, X, fmt = ['%2.9f', '%2.9f'], delimiter=' ', newline='\n')
         
        dist = Generator(input=self.distgen_file)#, verbose=True)
        #dist.input['start']['MTE']['value'] = MTE # Should be in YAML
        dist.input['r_dist']['max_r']['value'] = radius #mm based on scaling in call script 
        dist.run()
        particles = dist.particles
        particles.write_opal('opal_emitted.txt', dist_type= 'emitted')
        self.dist = dist
        return dist


    def make_ssnl_dist(self,x):
        """Run distgen with output from ssnl code"""
        # Load laser inputs
        if len(self.xscale)==1:
            rscale  = 1
            gscale  = 1
            sfscale = 1
        else:
            # Load scales
            rindex  = np.where(self.xscale['name']=='RADIUS')
            gindex  = np.where(self.xscale['name']=='GDD')
            sfindex = np.where(self.xscale['name']=='SF')
            rscale  = self.xscale['scale'][rindex][0]
            gscale  = self.xscale['scale'][gindex][0]
            sfscale = self.xscale['scale'][sfindex][0]

        #Run python ssnl to get longitudinal profile
        import ssnl 
        a   = ssnl.SSNL()
        u   = ssnl.UNITS()
        gdd = x[gindex][0]*gscale*1000
        sf  = x[sfindex][0]*sfscale

        a.set_default(gdd, sf, 1024, 246)
        a.genEqns()
        a.genGrids()
        a.genFields()
        a.propagate()
        t, uv_intensity = a.generate_uv(self.laser_filter) #filter assumed in nm
        # Filter green profile, get UV profile 
        #green = a.eField['freq'][3][-1,:]
        #filt = np.exp( -( (a.lists['lambda'][2,:] - a.lams[2]*u.nm) / (self.laser_filter*u.nm) )**4 );
        #green_filtered = green* filt;
        #green_ifft     = ssnl.ifft(green_filtered)
        #green_intensity= abs(green_ifft**2)
        #uv_intensity   = green_intensity**2
        #
        ## Time distribution in ps
        #t = np.arange(0,len(uv_intensity))*a.grids['dt']*10**12
        #t = t-max(t)/2

        X  = np.zeros((len(t),2))
        X[:,0] = t
        X[:,1] = uv_intensity
        #print('sum UV intensity', sum(uv_intensity))
        with open('tdist_file_ssnl.dat', 'wb') as f:
            np.savetxt(f, X, fmt = ['%2.9f', '%2.9f'], delimiter=' ', newline='\n')

        dist = Generator(input=self.distgen_file)#, verbose=True)
        dist.input['r_dist']['max_r']['value'] = rscale*x[rindex][0] 
        dist.input['t_dist']['file'] = os.getcwd() + '/tdist_file_ssnl.dat'
        try: # making distribution 
            dist.run()
            particles = dist.particles
            particles.write_opal('opal_emitted_ssnl.txt', dist_type= 'emitted')
            self.dist = dist
        except Exception as e:
            print(e) # a None value will be held in self.dist 
            print('Making distribution with distgen failed')
            print('GDD:', gdd)
            print('SF:', sf)
        return self.dist


    def make_sim_file(self,x):
        """Make opal input files """
        # Linking magnet and RF files
        #linkDirectory(self.fieldmap_path, 'Fieldmap')
                #make data file
        opaldict = {}
        fp = open(self.data_file, "r")
        for line in fp:
            if not line == "\n":
                li = line.strip()
                # ignore outcommented lines
                if not li.startswith("#"):
                    # cut off comments at the end of the line
                    aline = line.split("#")[0]
                    # the name-value pairs are separated by whitespace
                    name, val = aline.split()
                    #import pdb; pdb.set_trace()
                    #print('x', x)
                    if name in self.dvar_keys:
                        #print('name', name)
                        name_index = np.where(self.xscale['name']==name)
                        #print('name_index', name_index)
                        name_scale = self.xscale['scale'][name_index][0]
                        #print('name_scale', name_scale)
                        #print('x[name_index]', x[name_index])
                        name_val   = name_scale*x[name_index][0]
                        #try:
                        #    name_val   = name_scale*x[name_index][0]
                        #except:
                        #    name_val   = name_scale*x[0][name_index]
                        opaldict[name.rstrip()] = str(name_val)
                    else:
                        opaldict[name.rstrip()] = val.lstrip().rstrip()
        fp.close()

        self.opal_input_file = self.basefile_name+'.in'
        # Read in the data and tmpl files
        filedata = None
        with open(self.template_file, 'r') as file:
            filedata = file.read()
        # do the replacements in the templatefile
        for s, value in opaldict.items():
            # Replace the target string
            filedata = filedata.replace('_'+s+'_', str(value))
        # Write the file out again
        with open(self.opal_input_file, 'w') as file:
            file.write(filedata)
        return #opal_input

    def run_sim(self, x):
        """Run distgen/opal or opal, get calc_status"""
        # Make unique distribution
        if 'GDD' in self.xscale['name']:    
            distribution = self.make_ssnl_dist(x) 
        elif 'FWHM' in self.xscale['name']:
            distribution = self.make_gaussian_dist(x)
       
        # Make input file
        self.make_sim_file(x)
        # Starting simulation with xvals
        exctr = Executor.executor
        task  = exctr.submit(calc_type='sim', num_procs=self.cores, app_args= self.opal_input_file, #machinefile='machinefile',
                             stdout='out.txt', stderr='err.txt', extra_args='--exact -u --mem-per-cpu=4G') 
        #'--exclusive') #"--exact -u -bind-to core")#, dry_run=True) 
       
        # Waiting for task to finish
        poll_interval = 1
        while not task.finished :
            if task.runtime > self.time_limit:
                task.kill()
            else:
                time.sleep(poll_interval)
                task.poll()   
       
        if task.finished:
            if task.state == 'FINISHED':
                print("Task {} completed".format(task.name))
                calc_status = WORKER_DONE
                #if read_last_line(filepath) == "kill":
                #    print("Warning: Task complete but marked bad (kill flag in forces.stat)")
            elif task.state == 'FAILED':
                print("Warning: Task {} failed: Error code {}".format(task.name, task.errcode))
                calc_status = TASK_FAILED
            elif task.state == 'USER_KILLED':
                print("Warning: Task {} has been killed".format(task.name))
                calc_status = WORKER_KILL
            else:
                print("Warning: Task {} in unknown state {}. Error code {}".format(task.name, task.state, task.errcode))
        
        time.sleep(0.25) # Small buffer to guarantee data has been written
        self.calc_status = calc_status
        return #calc_status       


    def parse_stat(self):
        """load OPAL stat data"""
        nRows = None
        try:
            #getting number of rows in header
            nRows  = parseHeader(self.stat_file)
        except Exception as e:
            print(e)
            print('Unable to load data here:', os.getcwd())

        if nRows:
            alldata = {}
            #File exists, header there
            file_data = np.loadtxt(self.stat_file, skiprows=nRows)             
            for i,name in enumerate(STAT_NAMES):
                alldata[name] = file_data[:,i]
            #File reached correct z location
            if alldata['s'][-1] > self.zstop*0.995: # with in 0.5% of end location
                dlen = len(alldata['s'])
                for outname in self.data_keys:
                    self.output[outname][0] = alldata[outname][-1]
                    self.output[outname +'_long'][0,0:dlen] = alldata[outname]
            else:
                print('Warning, data did not reach specified z location. Not returning data for this run:', os.getcwd())
        else:
            print('Warning, no data to returned for this run:', os.getcwd())
        return #output 
        
    #def parse_h5(self):
    #    pass #ADD ME
    #    return 


    def penalty_check(self):
        """Add penalties for bad beam behavior"""

        #try:
        if self.zobj == None:
            zind = -1
        else:
            # in case z is slightly lower
            z = self.zobj-0.0005
            #Finding index of z location
            zind = np.argmax(self.output['s_long'][0]>z)
        #number of particles
        numpart = np.trim_zeros(self.output['numParticles_long'][0], trim='b')[zind]   
        self.penalty    = (self.max_particles - numpart) / self.penalty_scale 
        #print('numpart', numpart)
        #print('penalty:', self.penalty)
        return 

    def get_obj_vals(self):
        """ 
        Return the stat value near z value to zobj.
    
        output   = dict containing all stat information
        obj_keys = stat keys to save and return 
        zobj     = float, z location where you want objectives
        """
        if self.zobj == None:
            zind = -1
        else:
            # in case z is slightly lower
            z = self.zobj-0.0005
            #Finding index of z location
            zind  = np.argmax(self.output['s_long']>z)
        
        for key in self.obj_keys:
            #Finding objectives at specified z location
            self.raw_obj_vals[key] = np.trim_zeros(self.output[key+'_long'][0], trim='b')[zind]
        return #objs

    def scale_obj_vals(self): 
        """Return scaled objectives"""
        obj_names   = self.obj_scale['name']
        # Using obj_scale['name'] ensures same order each time
        for i, key in enumerate(obj_names):
            obj_index      = np.where(self.obj_scale['name']==key)  
            denominator    = self.obj_scale['ub'][obj_index][0] - self.obj_scale['lb'][obj_index][0]
            self.scaled_obj_vals[i] = (self.raw_obj_vals[key]-self.obj_scale['lb'][obj_index][0])/denominator
        return 


def xrms_sum(output, z1):
    """Checking smooth decrease in xrms"""
    xsum      = 0
    zvalues   = output['s_long'][0]
    xrms      = output['rms_x_long'][0]
    front     = zvalues<z1
    zindex1   = np.argmax(xrms[front])
    print('z at argmax', output['s_long'][0][zindex1])
    #print('z area of interest', np.trim_zeros(zvalues[zindex1:], 'b'))
    xrms_test = np.trim_zeros(xrms[zindex1:], 'b')
    for i in range(0,len(xrms_test)-1):
        print('xrms check numbers')
        print(i, xrms_test[i+1], xrms_test[i], xrms_test[i+1]-xrms_test[i])
        xsum =+ max(0, xrms_test[i+1]-xrms_test[i])
    #Scale xrms
    xsum = xsum*10**4
    print('xsum for this simulation:', xsum)
    return xsum

def load_data(osim, sim_specs):
    osim.parse_stat()     # load output
    osim.penalty_check()  # load cost / do penalty check
    osim.get_obj_vals()   # get raw obj values
    osim.scale_obj_vals() # scale obj values
    fvals = np.zeros(len(sim_specs['user']['objective_scales']))
    # fill in objective values 
    for i, key in enumerate(sim_specs['user']['key_dict']['objective_keys']):
        fvals[i] = osim.scaled_obj_vals[i]+osim.penalty

    try: # addition for vtmop paper 
        emit_scaled = osim.scaled_obj_vals[0]
        if emit_scaled > 0.3:
            #print('scaled emit is', emit_scaled)
            #print('raw emit is', osim.raw_obj_vals['emit_x'])
            #print('lost particles penalty', osim.penalty)
            #print('emit penalty', emit_scaled - 0.3)
            fvals[:] = fvals[:] + emit_scaled - 0.3
    except:
       print('lost particles penalty', osim.penalty)
       pass
 
    return fvals

def opal_aposmm(H, persis_info, sim_specs, _):
    """aposmm optimization algorithm w/ opal sim"""
    x = H['x'][0]
    fvals = np.zeros(len(sim_specs['user']['objective_scales']))
    osim  = LibeOpal(sim_specs)
    w     = osim.weights
    osim.run_sim(x)
    print('weights', w)

    try: # Try to load data
        print('input file:', osim.opal_input_file)
        fvals = load_data(osim, sim_specs)
    except Exception as e:
        print('No data loaded, returned error:', e)
        fvals[:] = sim_specs['user']['sim_particles']/sim_specs['user']['penalty_scale']


    print('f', fvals)
    print('w*fvals', w*fvals) 
    output['f'][0] = sum(fvals*w)
    return osim.output, persis_info, osim.calc_status 


def opal_vtmop(H, persis_info, sim_specs, _):
    """vtmop optimization algorithm w/ opal sim"""
    x     = H['x'][0]
    fvals = np.zeros(len(sim_specs['user']['objective_scales']))
    osim  = LibeOpal(sim_specs)
    osim.run_sim(x)
    
    try: # try to load sim data
        print('input file:', osim.opal_input_file)
        fvals = load_data(osim, sim_specs)
    except Exception as e:
        print('No data loaded, returned error:', e)
        fvals[:] = sim_specs['user']['sim_particles']/sim_specs['user']['penalty_scale']

    osim.output['f'][0] = fvals
    print('f objectives for current sim:', osim.output['f'][0])
    sys.stdout.flush()
    return osim.output, persis_info, osim.calc_status

    return osim        

def opal_deap(H, persis_info, sim_specs, _):
    """deap optimization algorithm w/ opal sim"""
    x     = H['individual'][0]
    fvals = np.zeros(len(sim_specs['user']['objective_scales']))
    osim  = LibeOpal(sim_specs)
    osim.run_sim(x)

    try:
        print('input file:', osim.opal_input_file)
        fvals = load_data(osim, sim_specs)
        #for i, key in enumerate(sim_specs['user']['key_dict']['objective_keys']): 
            #fvals[i]  = osim.raw_obj_vals[key]+osim.penalty
            #fvals[i] = osim.scaled_obj_vals[i]+osim.penalty
    except Exception as e:
        print('No data loaded, returned error:', e)
        #fvals[:] = osim.penalty
        fvals[:] = sim_specs['user']['sim_particles']/sim_specs['user']['penalty_scale']
       
        
    osim.output['fitness_values'][0] = fvals 
    print('f objectives for current sim:', osim.output['fitness_values'][0])
    sys.stdout.flush()
    return osim.output, persis_info, osim.calc_status


def opal_sample(H, persis_info, sim_specs, _):
    """random sample with opal sim"""
    osim  = LibeOpal(sim_specs)
    osim.run_sim(H['x'][0])
    try:
        fvals = load_data(osim, sim_specs)
    except Exception as e:
        print('No data loaded, returned error:', e)
        fvals = np.zeros(len(sim_specs['user']['objective_scales']))
        fvals[:] = sim_specs['user']['sim_particles']/sim_specs['user']['penalty_scale']
 
    osim.output['f'][0] = fvals # unweight fvals w/ penalty
    return osim.output, persis_info, osim.calc_status

def opal_nlopt(H, persis_info, sim_specs, _):
    """single objective optimizations"""
    osim = LibeOpal(sim_specs)
    osim.run_sim(H['x'][0])
    try:
        load_data(osim)
        osim.scale_obj_vals()
        osim.output['f']  = osim.scaled_obj_vals+osim.penalty
        print('output', osim.output['f'])
    except Exception as e:
        print('No data loaded, error:', e)
        osim.output['f']  = sim_specs['user']['sim_particles']/sim_specs['user']['penalty_scale']
    return osim.output, persis_info, osim.calc_status
