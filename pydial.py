#! /usr/bin/env python

###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2017
# Cambridge University Engineering Department Dialogue Systems Group
#
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################

import os
from scriptine import run, path, log, command
import re
import numpy as np

# Uncomment for mac os users
#import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Agg')

import matplotlib.pyplot as plt

#Uncomment for 4k screens
# matplotlib.rcParams.update({'font.size': 22})

# PyDial modules
import Simulate
import Texthub
from utils import Settings
from utils import ContextLogger
from ontology import Ontology
import utils.ContextLogger as clog
import pprint
pp = pprint.PrettyPrinter(indent=4)

logger = None
tracedialog = 2
policy_dir = ""
conf_dir = ""
log_dir = ""
logfile = ""

gnumtrainbatches = 0
gtraindialogsperbatch = 0
gnumbatchtestdialogs = 0
gnumtestdialogs = 0
gtrainerrorrate = 0
gtesterrorrate = 0
gtrainsourceiteration = 0
gtesteverybatch = False

gpscale = 1

gplotnum=1

isSingleDomain = True
taskID = ""
domain = ""
domains = []
policytype = ""

def help_command():
    """ Provide an overview of pydial functionality
    """
    print "\n pydial - command line interface to PyDial"
    print '""""""""""""""""""""""""""""""""""""""""""""'
    print ' o Runs simulator to train and test policies'
    print ' o Plots learning rates and performance vs error rate'
    print ' o Runs texthub in multi-domain chat mode\n'
    print 'Basic usage:'
    print '  a) Make pydial.py executable and add a symbolic link to it (eg pydial) from your'
    print '     local bin directory.  Create a directory called ID and cd into it.\n'
    print "  b) create a config file and add an exec_config section eg:\n"
    print '     [exec_config]'
    print '     domain = CamRestaurants     # specific train/test domain'
    print '     policytype = gp             # type of policy to train/test'
    print '     configdir = cfgdir          # folder to store configs'
    print '     logfiledir = logdir         # folder to store logfiles'
    print '     numtrainbatches = 2         # num training batches (iterations)'
    print '     traindialogsperbatch = 10   # num dialogs per batch'
    print '     numbatchtestdialogs =  20   # num dialogs to eval each batch'
    print '     trainsourceiteration = 0    # index of initial source policy to update'
    print '     testiteration = 1           # policy iteration to test'
    print '     numtestdialogs =  10        # num dialogs per test'
    print '     trainerrorrate = 0          # train error rate in %'
    print '     testerrorrate  = 0          # test error rate in %'
    print '     testeverybatch = True       # enable batch testing\n'
    print '     by convention the config file name for training and testing should be of the'
    print '     form ID-policytype-domain.cfg where ID is a user-defined id.'
    print '     (There is more detail on naming conventions below.)'
    print '     Also unless the current directory is the same as the PyDial root'
    print '     make sure that [GENERAL]root points to root of the PyDial source tree.\n'
    print '  c) to train a policy as specified in the config file, type'
    print '       > pydial train config'
    print '     if trainsourceiteration=0 this creates a new policy in n batches where'
    print '     n=numtrainbatches, otherwise an existing policy is trained further.\n'
    print '  d) to test a policy as specified in the config file, type'
    print '       > pydial test config\n'
    print '     texthub.py can be invoked to interact with a policy from the keyboard by:'
    print '       > pydial chat config'
    print '     Note that train and test must refer to a specific domain as per [exec_config] domain'
    print '     whereas chat mode can specify multiple domains via the [GENERAL]domains variable.\n'
    print '  e) for convenience, many config parameters can be overridden on the command line, eg'
    print '       > pydial train config --trainerrorrate=20'
    print '       > pydial test config --iteration=4 --trainerrorrate=20 --testerrorrate=50'
    print '     to train a policy at 20% error rate and test the 4th iteration at 50% error rate.'
    print '     A range of test error rates can be specified as a triple (stErr,enErr,stepSize), eg'
    print "       > pydial test config --iteration=4 --trainerrorrate=20 --testerrorrate='(0,50,10)'"
    print '     to test a policy at 0%, 10%, 20%, 30%, 40%, and 50% error rates.\n'
    print '  f) logfiles for each train/test run are stored in logfiledir.'
    print '     The plot command scans one or more logfiles and extract info to plot eg'
    print '       > pydial plot logdir/*train*'
    print '     Setting the option --printtab, also tabulates the performance data.\n'
    print '  All policy information is stored in the policydir specified in the corresponding '
    print '  config file section with name [policy_domain]. Since pydial overrides some config'
    print '  params, the actual configs used for each run are recorded in configdir.\n'
    print '  Derived file naming convention:'
    print "     Policyname: ID-poltype-domain-TrainErrRate               eg S0-gp-CamRestaurants-20"
    print "     Policy: ID-poltype-domain-TrainErrRate.Iteration         eg S0-gp-CamRestaurants-20.3"
    print "     Policyfile: ID-poltype-domain-TrainErrRate.Iteration.ext eg S0-gp-CamRestaurants-20.3.dct"
    print "     TrainLogfiles: PolicyName.IterationRange.train.log       eg S0-gp-CamRestaurants-20.1-3.train.log"
    print "     EvalLogfiles:  Policy.eval.ErrorRange.eval.log           eg S0-gp-CamRestaurants-20.3.eval.00-50.log\n"
    print "To get further help:"
    print "  pydial             list of available commands"
    print "  pydial help        this overview"
    print "  pydial cmd --help  help for a specific command\n"

def conventionCheck(name):
    global taskID,domain,policytype
    try:
        if name.find('-')<0:
            raise Exception('no separators')
        (taskID,p,d)=name.split('-')
        if p != policytype:
            raise Exception('policytype != config param')
        if d != domain:
            raise Exception('domain name != config param')
    except Exception as x:
        log.warn("Non-standard config name [%s] (preferred format ID-policytype-domain.cfg)", x.args[0])

def getConfigId(configFileName):
    i = configFileName.rfind('.')
    if i<0 or configFileName[i+1:]!='cfg':
        print ("Config file %s does not have required .cfg extension" % configFileName)
        exit(0)
    cfg = path(configFileName)
    if not cfg.isfile():
        print ("Config file %s does not exist" % configFileName)
        exit(0)
    id = configFileName[:i]
    j = id.rfind('/')
    if j>=0: id = id[j+1:]
    return id

def getOptionalConfigVar(configvarname, default='', section='exec_config'):
    value = default
    if Settings.config.has_option(section, configvarname):
        value = Settings.config.get(section, configvarname)
    return value

def getRequiredDirectory(directoryname, section='exec_config'):
    assert Settings.config.has_option(section, directoryname),\
        "Value {} in section {} is missing.".format(directoryname, section)
    dir = Settings.config.get(section, directoryname)
    if dir[-1] != '/': dir = dir+'/'
    return dir

def getOptionalConfigInt(configvarname, default='0',section='exec_config'):
    value = default
    if Settings.config.has_option(section, configvarname):
        value = Settings.config.getint(section, configvarname)
    return value

def getOptionalConfigBool(configvarname, default='False', section='exec_config'):
    value = default
    if Settings.config.has_option(section, configvarname):
        value = Settings.config.getboolean(section, configvarname)
    return value

def initialise(configId,config_file, seed, mode, trainerrorrate=None, trainsourceiteration=None,
               numtrainbatches=None, traindialogsperbatch=None, numtestdialogs=None,
               testerrorrate=None, testenderrorrate=None, iteration=None):
    global logger, logfile, traceDialog, isSingleDomain
    global policy_dir, conf_dir, log_dir
    global gnumtrainbatches, gtraindialogsperbatch, gnumbatchtestdialogs, gnumtestdialogs
    global gtrainerrorrate, gtesterrorrate, gtrainsourceiteration
    global taskID, domain, domains, policytype, gtesteverybatch, gpscale
    global gdeleteprevpolicy


    if seed:
        seed = int(seed)
    seed = Settings.init(config_file, seed)
    taskID='ID'

    isSingleDomain = getOptionalConfigBool("isSingleDomain", isSingleDomain, "GENERAL")
    traceDialog    = getOptionalConfigInt("tracedialog", tracedialog, "GENERAL")
    domain         = getOptionalConfigVar("domain")
    if isSingleDomain:
        policytype = getOptionalConfigVar('policytype', policytype, 'policy_' + domain)
        conventionCheck(configId)
    else:
        domains = getOptionalConfigVar("domains", "", "GENERAL").split(',')
        policytypes = {}
        for domain in domains:
            policytypes[domain] = getOptionalConfigVar('policytype', policytype, 'policy_' + domain)

    # if gp, make sure to save required scale before potentially overriding
    if isSingleDomain:
        if policytype == "gp":
            gpscale = Settings.config.getint("gpsarsa_" + domain, "scale")
        else:
            gpscales = {}
            for domain in domains:
                if policytypes[domain] == "gp":
                    gpscales[domain] = Settings.config.getint("gpsarsa_" + domain, "scale")

    # Get required folders and create if necessary
    log_dir    = getRequiredDirectory("logfiledir")
    conf_dir   = getRequiredDirectory("configdir")
    if isSingleDomain:
        if policytype != 'hdc':
            policy_dir = getRequiredDirectory("policydir","policy_"+domain)
            pd = path(policy_dir)
            if not pd.isdir():
                print "Policy dir %s does not exist, creating it" % policy_dir
                pd.mkdir()
    else:
        for domain in domains:
            if policytypes[domain] != 'hdc':
                policy_dir = getRequiredDirectory("policydir", "policy_" + domain)
                pd = path(policy_dir)
                if not pd.isdir():
                    print "Policy dir %s does not exist, creating it" % policy_dir
                    pd.mkdir()

    cd = path(conf_dir)
    if not cd.isdir():
        print "Config dir %s does not exist, creating it" % conf_dir
        cd.mkdir()
    ld = path(log_dir)
    if not ld.isdir():
        print "Log dir %s does not exist, creating it" % log_dir
        ld.mkdir()


    # optional config settings
    if numtrainbatches:
        gnumtrainbatches = int(numtrainbatches)
    else:
        gnumtrainbatches = getOptionalConfigInt("numtrainbatches",1)
    if traindialogsperbatch:
        gtraindialogsperbatch = int(traindialogsperbatch)
    else:
        gtraindialogsperbatch = getOptionalConfigInt("traindialogsperbatch",100)
    if trainerrorrate:
        gtrainerrorrate = int(trainerrorrate)
    else:
        gtrainerrorrate = getOptionalConfigInt("trainerrorrate", 0)
    if testerrorrate:
        gtesterrorrate = int(testerrorrate)
    else:
        gtesterrorrate = getOptionalConfigInt("testerrorrate",0)
    if trainsourceiteration:
        gtrainsourceiteration = int(trainsourceiteration)
    else:
        gtrainsourceiteration = getOptionalConfigInt("trainsourceiteration",0)
    if numtestdialogs:
        gnumtestdialogs = int(numtestdialogs)
    else:
        gnumtestdialogs = getOptionalConfigInt("numtestdialogs", 50)

    gnumbatchtestdialogs = getOptionalConfigInt("numbatchtestdialogs", 20)
    gtesteverybatch = getOptionalConfigBool("testeverybatch",True)
    gdeleteprevpolicy = getOptionalConfigBool("deleteprevpolicy", False)
    if mode=="train":
        if gnumtrainbatches>1:
            enditeration = gtrainsourceiteration+gnumtrainbatches
            logfile = "%s-%02d.%d-%d.train.log" % (configId,gtrainerrorrate,gtrainsourceiteration+1,enditeration)
        else:
            logfile = "%s-%02d.%d.train.log" % (configId, gtrainerrorrate, gtrainsourceiteration + 1)
    elif mode=="eval":
        if testenderrorrate:
            logfile = "%s-%02d.%d.eval.%02d-%02d.log" % (configId,gtrainerrorrate,iteration,
                                                         gtesterrorrate,testenderrorrate)
        else:
            logfile = "%s-%02d.%d.eval.%02d.log" % (configId, gtrainerrorrate, iteration, gtesterrorrate)
    elif mode=="chat":
        logfile = "%s-%02d.%d.chat.log" % (configId, gtrainerrorrate, gtrainsourceiteration)
    else:
        print "Unknown initialisation mode:",mode
        exit(0)

    Settings.config.set("logging", "file", log_dir + logfile)
    ContextLogger.createLoggingHandlers(config=Settings.config)
    logger = ContextLogger.getLogger('')
    Ontology.init_global_ontology()
    if Settings.root=='':
        Settings.root = os.getcwd()
    logger.info("Seed = %d", seed)
    logger.info("Root = %s", Settings.root)

def setupPolicy(domain, configId, trainerr, source_iteration,target_iteration):
    policy_section = "policy_" + domain
    inpolicyfile = "%s-%02d.%d" % (configId, trainerr, source_iteration)
    outpolicyfile = "%s-%02d.%d" % (configId, trainerr, target_iteration)
    if isSingleDomain:
        Settings.config.set(policy_section, "inpolicyfile", policy_dir + inpolicyfile)
        Settings.config.set(policy_section, "outpolicyfile", policy_dir + outpolicyfile)
    else:
        multi_policy_dir = policy_dir + domain
        pd = path(multi_policy_dir)
        if not pd.isdir():
            print "Policy dir %s does not exist, creating it" % multi_policy_dir
            pd.mkdir()
        Settings.config.set(policy_section, "inpolicyfile", multi_policy_dir + inpolicyfile)
        Settings.config.set(policy_section, "outpolicyfile", multi_policy_dir + outpolicyfile)
    return (inpolicyfile,outpolicyfile)


def trainBatch(domain, configId, trainerr, ndialogs, source_iteration):
    if isSingleDomain:
        (inpolicy, outpolicy) = setupPolicy(domain, configId, trainerr, source_iteration, source_iteration + 1)
        mess = "*** Training Iteration %s->%s: iter=%d, error-rate=%d, num-dialogs=%d ***" % (
            inpolicy, outpolicy, source_iteration, trainerr, ndialogs)
        if tracedialog > 0: print mess
        logger.results(mess)
        # make sure that learning is switched on
        Settings.config.set("policy_" + domain, "learning", 'True')
        # if gp, make sure to reset scale to config setting
        if policytype == "gp":
            Settings.config.set("gpsarsa_" + domain, "scale", str(gpscale))
        # Define the config file for this iteration
        confsavefile = conf_dir + outpolicy + ".train.cfg"
    else:
        mess = "*** Training Iteration: iter=%d, error-rate=%d, num-dialogs=%d ***" % (
            source_iteration, trainerr, ndialogs)
        if tracedialog > 0: print mess
        logger.results(mess)
        for dom in domain:
            setupPolicy(dom, configId, trainerr, source_iteration, source_iteration + 1)
            # make sure that learning is switched on
            Settings.config.set("policy_" + dom, "learning", 'True')
            # if gp, make sure to reset scale to config setting
            if policytype == "gp":
                Settings.config.set("gpsarsa_" + dom, "scale", str(gpscale))
        # Define the config file for this iteration
        multipolicy = "%s-%02d.%d" % (configId, trainerr, source_iteration + 1)
        confsavefile = conf_dir + multipolicy + ".train.cfg"

    # Save the config file for this iteration
    cf = open(confsavefile, 'w')
    Settings.config.write(cf)
    error = float(trainerr) / 100.0
    # run the system
    simulator = Simulate.SimulationSystem(error_rate=error)
    simulator.run_dialogs(ndialogs)
    if gdeleteprevpolicy:
        if isSingleDomain:
            if inpolicy[-1] != '0':
                print 'rm {}/*{}*'.format(Settings.config.get('policy_{}'.format(domain),'policydir'),inpolicy)
                os.system('rm {}/*{}*'.format(Settings.config.get('policy_{}'.format(domain),'policydir'),inpolicy))



def setEvalConfig(domain, configId, evalerr, ndialogs, iteration):
    (_, policy) = setupPolicy(domain, configId, gtrainerrorrate, iteration, iteration)
    if isSingleDomain:
        mess = "*** Evaluating %s: error-rate=%d num-dialogs=%d ***" % (policy, evalerr, ndialogs)
    else:
        mess = "*** Evaluating %s: error-rate=%d num-dialogs=%d ***" % (policy.replace('Multidomain', domain),
                                                                        evalerr, ndialogs)
    if tracedialog > 0: print mess
    logger.results(mess)
    # make sure that learning is switched off
    Settings.config.set("policy_" + domain, "learning", 'False')
    # if gp, make sure to reset scale to 1 for evaluation
    if policytype == "gp":
        Settings.config.set("gpsarsa_" + domain, "scale", "1")
    # Save a copy of config file
    confsavefile = conf_dir + "%s.eval.%02d.cfg" % (policy, evalerr)
    cf = open(confsavefile, 'w')
    Settings.config.write(cf)

def evalPolicy(domain, configId, evalerr, ndialogs, iteration):
    if isSingleDomain:
        setEvalConfig(domain, configId, evalerr, ndialogs, iteration)
    else:
        for dom in domains:
            setEvalConfig(dom, configId, evalerr, ndialogs, iteration)

    error = float(evalerr) / 100.0
    # finally run the system
    simulator = Simulate.SimulationSystem(error_rate=error)
    simulator.run_dialogs(ndialogs)

def getIntParam(line,key):
    m = re.search(" %s *= *(\d+)" % (key), line)
    if m==None:
        print "Cant find int %s in %s" % (key,line)
        exit(0)
    return int(m.group(1))

def getFloatRange(line,key):
    m = re.search(" %s *= *(\-?\d+\.\d+) *\+- *(\d+\.\d+)" % (key), line)
    if m==None:
        print "Cant find float %s in %s" % (key,line)
        exit(0)
    return (float(m.group(1)),float(m.group(2)))

def getDomainFromLog(l):
    return l[l.find('ontologies'):][l[l.find('ontologies'):].find('/') + 1:l[l.find('ontologies'):].find('-')]

def extractEvalData(lines):
    evalData = {}
    training = False
    domain_list = []
    cur_domain = None
    for l in lines:
        if l.find('Loading ontology') >= 0:
            # get the list of domains from the log by reading the lines where the ontologies are loaded
            domain = getDomainFromLog(l)
            domain_list.append(domain)
            evalData[domain] = {}
        if l.find('*** Training Iteration')>=0:
            iteration = getIntParam(l,'iter')+1
            if iteration in evalData.keys():
                print "Duplicate iteration %d" % iteration
                exit(0)
            for domain in domain_list:
                evalData[domain][iteration] = {}
                evalData[domain][iteration]['erate'] = getIntParam(l,'error-rate')
                evalData[domain][iteration]['ndialogs'] = getIntParam(l,'num-dialogs')
            training = True
        elif l.find('*** Evaluating')>=0 and not training:
            erate = getIntParam(l,'error-rate')
            ll = l[l.find('*** Evaluating') + len('*** Evaluating')+1:]
            (ll,x) = ll.split(':')
            for domain in domain_list:
                if domain in ll:
                    evalData[domain][erate] = {}
                    evalData[domain][erate]['policy'] = ll
                    evalData[domain][erate]['ndialogs'] = getIntParam(l,'num-dialogs')
        elif l.find('Results for domain:')>=0:
            cur_domain = l.split('Results for domain:')[1].split('--')[0].strip()
        elif l.find('Average reward')>=0:
            if training:
                evalData[cur_domain][iteration]['reward'] = getFloatRange(l,'Average reward')
            else:
                evalData[cur_domain][erate]['reward'] = getFloatRange(l,'Average reward')
        elif l.find('Average success')>=0:
            if training:
                evalData[cur_domain][iteration]['success'] = getFloatRange(l, 'Average success')
            else:
                evalData[cur_domain][erate]['success'] = getFloatRange(l,'Average success')

        elif l.find('Average turns')>=0:
            if training:
                evalData[cur_domain][iteration]['turns'] = getFloatRange(l, 'Average turns')
            else:
                evalData[cur_domain][erate]['turns'] = getFloatRange(l,'Average turns')
    return evalData

def plotTrain(dname,rtab,stab,block=True,saveplot=False):
    global gplotnum
    policylist = sorted(rtab.keys())
    ncurves = len(policylist)
    plt.figure(gplotnum)
    gplotnum += 1
    for policy in policylist:
        tab = rtab[policy]
        plt.subplot(211)
        plt.errorbar(tab['x'],tab['y'],yerr=tab['var'],label=policy)
        tab = stab[policy]
        plt.subplot(212)
        plt.errorbar(tab['x'],tab['y'],yerr=tab['var'],label=policy)
    plt.subplot(211)
    plt.grid()
    plt.legend(loc='lower right',fontsize=12-ncurves)
    plt.title(dname+" Performance vs Num Train Dialogs")
    plt.ylabel('Reward')
    plt.subplot(212)
    plt.grid()
    plt.legend(loc='lower right',fontsize=12-ncurves)
    plt.xlabel('Num Dialogs')
    plt.ylabel('Success')
    if saveplot:
        if not os.path.exists('_plots'):
            os.mkdir('_plots')
        plt.savefig('_plots/' + dname + '.png', bbox_inches='tight')
    else:
        plt.show(block=block)

def plotTest(dname,rtab,stab,block=True,saveplot=False):
    global gplotnum
    policylist = sorted(rtab.keys())
    ncurves = len(policylist)
    plt.figure(gplotnum)
    gplotnum += 1
    for policy in policylist:
        tab = rtab[policy]
        plt.subplot(211)
        plt.errorbar(tab['x'],tab['y'],yerr=tab['var'],label=policy)
        tab = stab[policy]
        plt.subplot(212)
        plt.errorbar(tab['x'],tab['y'],yerr=tab['var'],label=policy)
    plt.subplot(211)
    plt.grid()
    plt.legend(loc='lower left',fontsize=12-ncurves)
    plt.title(dname+" Performance vs Error Rate")
    plt.ylabel('Reward')
    plt.subplot(212)
    plt.grid()
    plt.legend(loc='lower left',fontsize=12-ncurves)
    plt.xlabel('Error Rate')
    plt.ylabel('Success')
    plt.show(block=block)

def printTable(title, tab):
    firstrow = True
    policylist = sorted(tab.keys())
    for policy in policylist:
        xvals = tab[policy]['x']
        if firstrow:
            s = "%-20s" % title
            for i in range(0, len(xvals)): s += "%13d" % xvals[i]
            print s
            firstrow = False
        s = "%-18s :" % policy
        for i in range(0,len(xvals)):
            s+= "%6.1f +-%4.1f" % (tab[policy]['y'][i],tab[policy]['var'][i])
        print s
    print ""

def tabulateTrain(dataSet):
    #pp.pprint(dataSet)
    rtab = {}
    stab = {}
    ttab = {}
    oldx = []
    for policy in dataSet.keys():
        yvals = []
        xvals = []
        dialogsum = 0
        for iteration in dataSet[policy].keys():
            d = dataSet[policy][iteration]
            (yr, yrv) = d['reward']
            (ys, ysv) = d['success']
            (yt, ytv) = d['turns']
            ndialogs = d['ndialogs']
            dialogsum += ndialogs
            yvals.append((yr, yrv, ys, ysv, yt, ytv))
            xvals.append(dialogsum)
        yvals = [yy for (xx, yy) in sorted(zip(xvals, yvals))]
        x = [xx for (xx, yy) in sorted(zip(xvals, yvals))]
        if oldx != [] and oldx != x:
            print "Policy %s has inconsistent batch sizes" % policy
        oldx = x
        yrew = [yr for (yr, yrv, ys, ysv, yt, ytv) in yvals]
        yrerr = [yrv for (yr, yrv, ys, ysv, yt, ytv) in yvals]
        ysucc = [ys for (yr, yrv, ys, ysv, yt, ytv) in yvals]
        yserr = [ysv for (yr, yrv, ys, ysv, yt, ytv) in yvals]
        yturn = [yt for (yr, yrv, ys, ysv, yt, ytv) in yvals]
        yterr = [ytv for (yr, yrv, ys, ysv, yt, ytv) in yvals]
        if not (policy in rtab.keys()): rtab[policy] = {}
        rtab[policy]['y'] = yrew
        rtab[policy]['var'] = yrerr
        rtab[policy]['x'] = x
        if not (policy in stab.keys()): stab[policy] = {}
        stab[policy]['y'] = ysucc
        stab[policy]['var'] = yserr
        stab[policy]['x'] = x
        if not (policy in ttab.keys()): ttab[policy] = {}
        ttab[policy]['y'] = yturn
        ttab[policy]['var'] = yterr
        ttab[policy]['x'] = x
    # average results over seeds
    averaged_result_list = []
    for result in [rtab,stab,ttab]:
        averaged_result = {}
        n_seeds = {}
        for policy_key in result:
            if "seed" in policy_key:
                seed_n = policy_key[policy_key.find("seed"):]
                seed_n = seed_n.split('-')[0]
                general_policy_key = policy_key.replace(seed_n + '-', '')
            else:
                general_policy_key = policy_key
            if not general_policy_key in averaged_result:
                averaged_result[general_policy_key] = {}
                n_seeds[general_policy_key] = 1
            else:
                n_seeds[general_policy_key] += 1
            for key in result[policy_key]:
                if not key in averaged_result[general_policy_key]:
                    averaged_result[general_policy_key][key] = np.array(result[policy_key][key])
                else:
                    averaged_result[general_policy_key][key] += np.array(result[policy_key][key])
        for policy_key in averaged_result:
            for key in averaged_result[policy_key]:
                averaged_result[policy_key][key] = averaged_result[policy_key][key]/n_seeds[policy_key]
        averaged_result_list.append(averaged_result)

    return averaged_result_list

def tabulateTest(dataSet):
    #pp.pprint(dataSet)
    rtab = {}
    stab = {}
    ttab = {}
    oldx = []
    for policy in dataSet.keys():
        yvals = []
        xvals = []
        for erate in dataSet[policy].keys():
            d = dataSet[policy][erate]
            (yr,yrv) = d['reward']
            (ys,ysv) = d['success']
            (yt,ytv) = d['turns']
            yvals.append((yr,yrv,ys,ysv,yt,ytv))
            xvals.append(erate)
        yvals = [yy for (xx,yy) in sorted(zip(xvals,yvals))]
        x = [xx for (xx,yy) in sorted(zip(xvals,yvals))]
        if oldx != [] and oldx != x:
            print "Policy %s has inconsistent range of error rates" % policy
            exit(0)
        oldx = x
        yrew = [yr for (yr,yrv,ys,ysv,yt,ytv) in yvals]
        yrerr = [yrv for (yr,yrv,ys,ysv,yt,ytv) in yvals]
        ysucc = [ys for (yr,yrv,ys,ysv,yt,ytv) in yvals]
        yserr = [ysv for (yr,yrv,ys,ysv,yt,ytv) in yvals]
        yturn = [yt for (yr,yrv,ys,ysv,yt,ytv) in yvals]
        yterr = [ytv for (yr,yrv,ys,ysv,yt,ytv) in yvals]
        if not (policy in rtab.keys()): rtab[policy]={}
        rtab[policy]['y']=yrew
        rtab[policy]['var']=yrerr
        rtab[policy]['x']=x
        if not (policy in stab.keys()): stab[policy] = {}
        stab[policy]['y']=ysucc
        stab[policy]['var']=yserr
        stab[policy]['x']=x
        if not (policy in ttab.keys()): ttab[policy] = {}
        ttab[policy]['y']=yturn
        ttab[policy]['var']=yterr
        ttab[policy]['x']=x
    return (rtab,stab,ttab)

def train_command(configfile, seed=None, trainerrorrate=None,trainsourceiteration=None,
                  numtrainbatches=None,traindialogsperbatch=None):
    """ Train a policy according to the supplied configfile.
        Results are stored in the directories specified in the [exec_config] section of the config file.
        Optional parameters over-ride the corresponding config parameters of the same name.
    """
    try:
        if seed and seed.startswith('('):
            seeds = seed.replace('(','').replace(')','').split(',')
            for seed in seeds:
                pass #TODO implement multiseed training

        else:
            configId = getConfigId(configfile)
            if seed:
                seed = int(seed)
            initialise(configId,configfile,seed,"train",trainerrorrate=trainerrorrate,
                       trainsourceiteration=trainsourceiteration,numtrainbatches=numtrainbatches,
                       traindialogsperbatch=traindialogsperbatch)
            for i in range(gtrainsourceiteration,gtrainsourceiteration+gnumtrainbatches):
                if isSingleDomain:
                    trainBatch(domain, configId, gtrainerrorrate, gtraindialogsperbatch, i)
                else:
                    trainBatch(domains, configId, gtrainerrorrate, gtraindialogsperbatch, i)
                if gtesteverybatch and gnumbatchtestdialogs>0 and i+1 < gtrainsourceiteration+gnumtrainbatches:
                    if isSingleDomain:
                        evalPolicy(domain, configId, gtrainerrorrate, gnumbatchtestdialogs, i + 1)
                    else:
                        evalPolicy(domains, configId, gtrainerrorrate, gnumbatchtestdialogs, i + 1)
            if gnumbatchtestdialogs>0:
                if isSingleDomain:
                    evalPolicy(domain, configId, gtrainerrorrate, gnumbatchtestdialogs, i + 1)
                else:
                    evalPolicy(domains, configId, gtrainerrorrate, gnumbatchtestdialogs, i + 1)

            logger.results("*** Training complete - final policy is %s-%02d-%02d" % (configId,gtrainerrorrate,i+1))
    except clog.ExceptionRaisedByLogger:
        print "Command Aborted - see Log file for error:",logfile
        exit(0)
    except KeyboardInterrupt:
        print "\nCommand Aborted from Keyboard"

def test_command(configfile, iteration, seed=None, testerrorrate=None, trainerrorrate=None,
                 numtestdialogs=None):
    """ Test a specific policy iteration trained at a specific error rate according to the supplied configfile.
        Results are embedded in the logfile specified in the config file.
        Optional parameters over-ride the corresponding config parameters of the same name.
        The testerrorrate can also be specified as a triple (e1,e2,stepsize).  This will repeat the test
        over a range of error rates from e1 to e2.  NB the tuple must be quoted on the command line.
    """
    try:
        errStepping = False
        enErr = None
        if testerrorrate and testerrorrate[0] == '(':
            if testerrorrate[-1] != ')':
                print "Missing closing parenthesis in error range %s" % testerrorrate
                exit(0)
            errRange = eval(testerrorrate)
            if len(errRange) != 3:
                print "Ill-formed error range %s" % testerrorrate
                exit(0)
            (stErr, enErr, stepErr) = errRange
            if enErr < stErr or stepErr <= 0:
                print "Ill-formed test error range [%d,%d,%d]" % testerrorrate
                exit(0)
            errStepping = True
            testerrorrate = stErr
        i = int(iteration)
        if i < 1:
            print 'iteration must be > 0'
            exit(0)
        configId = getConfigId(configfile)
        if seed:
            seed = int(seed) + 100 # To have a different seed during training and testing
        initialise(configId, configfile, seed, "eval", iteration=i, testerrorrate=testerrorrate,
                   testenderrorrate=enErr, trainerrorrate=trainerrorrate,
                   numtestdialogs=numtestdialogs)
        policyname = "%s-%02d.%d" % (configId, gtrainerrorrate, i)
        poldirpath = path(policy_dir)
        if poldirpath.isdir():
            policyfiles = poldirpath.files()
            policynamelist = [p.namebase for p in policyfiles]
            if isSingleDomain:
                if policyname in policynamelist:
                    if errStepping:
                        while stErr <= enErr:
                            evalPolicy(domain, configId, stErr, gnumtestdialogs, i)
                            stErr += stepErr
                    else:
                        evalPolicy(domain, configId, gtesterrorrate, gnumtestdialogs, i)
                    logger.results("*** Testing complete - policy %s evaluated" % policyname)
                else:
                    print "Cannot find policy iteration %s in %s" % (policyname, policy_dir)
            else:
                allPolicyFiles = True
                for dom in domains:
                    multi_policyname = dom+policyname
                    if not multi_policyname in policynamelist:
                        print "Cannot find policy iteration %s in %s" % (multi_policyname, policy_dir)
                        allPolicyFiles = False
                if allPolicyFiles:
                    if errStepping:
                        while stErr <= enErr:
                            evalPolicy(domain, configId, stErr, gnumtestdialogs, i)
                            stErr += stepErr
                    else:
                        evalPolicy(domain, configId, gtesterrorrate, gnumtestdialogs, i)
                    logger.results("*** Testing complete - policy %s evaluated" % policyname)
        else:
            print "Policy folder %s does not exist" % policy_dir
    except clog.ExceptionRaisedByLogger:
        print "Command Aborted - see Log file for error:", logfile
        exit(0)
    except KeyboardInterrupt:
        print "\nCommand Aborted from Keyboard"

def plotTrainLogs(logfilelist,printtab,noplot,saveplot,datasetname,block=True):
    """
        Extract data from given log files and display.
    """
    try:
        resultset = {}
        ncurves = 0
        domains = None

        if len(logfilelist)<1:
            print "No log files specified"
            exit(0)
        for fname in logfilelist:
            fn = open(fname,"r")
            if fn:
                logName = path(fname).namebase
                i = logName.find('.')
                if i<0:
                    print "No index info in train log file name"
                    exit(0)
                curveName = logName[:i]
                if datasetname == '':
                    i = curveName.find('-')
                    if i>=0:
                        datasetname=curveName[:i]
                lines = fn.read().splitlines()
                results = extractEvalData(lines)
                npoints = len(results[results.keys()[0]])
                if npoints==0:
                    print "Log file %s has no plotable data" % fname
                else:
                    if len(resultset) == 0:
                        # the list of domains needs to be read from the logfile
                        domains = results.keys()
                        for domain in domains:
                            resultset[domain] = {}
                    else:
                        domains_1 = resultset.keys().sort()
                        domains_2 = results.keys().sort()
                        assert domains_1 == domains_2, 'The logfiles have different domains'
                    ncurves += 1
                    for domain in domains:
                        if curveName in resultset[domain].keys():
                            curve = resultset[domain][curveName]
                            for iteration in results.keys():
                                curve[iteration] = results[domain][iteration]
                        else:
                            resultset[domain][curveName] = results[domain]
            else:
                print("Cannot find logfile %s" % fname)
        if ncurves>0:
            average_results = [[],[],[]]
            for domain in domains:
                (rtab,stab,ttab) = tabulateTrain(resultset[domain])
                average_results[0].append(rtab)
                average_results[1].append(stab)
                average_results[2].append(ttab)
                if printtab:
                    print "\n%s-%s: Performance vs Num Dialogs\n" % (datasetname, domain)
                    printTable('Reward', rtab)
                    printTable('Success', stab)
                    printTable('Turns', ttab)
                if not noplot:
                    plotTrain(datasetname+'-'+domain,rtab,stab,block=block,saveplot=saveplot)
            # Print average for all domains
            av_rtab, av_stab, av_ttab = getAverageResults(average_results)
            plotTrain(datasetname+'-mean', av_rtab, av_stab, block=block,saveplot=saveplot)
        else:
            print("No plotable train data found")
    except clog.ExceptionRaisedByLogger:
        print "Command Aborted - see Log file for error:"

def getAverageResults(average_result_list):
    averaged_results = []
    for tab_list in average_result_list:
        n_domains = len(tab_list)
        tab_av_results = {}
        for domain_rtab in tab_list:
            for policy_key in domain_rtab:
                if not policy_key in tab_av_results:
                    tab_av_results[policy_key] = {}
                for key in domain_rtab[policy_key]:
                    if not key in tab_av_results[policy_key]:
                        if key == 'var':
                            tab_av_results[policy_key][key] = np.sqrt(np.array(domain_rtab[policy_key][key]))
                        else:
                            tab_av_results[policy_key][key] = np.array(domain_rtab[policy_key][key])
                    else:
                        if key == 'var':
                            tab_av_results[policy_key][key] += np.sqrt(np.array(domain_rtab[policy_key][key]))
                        else:
                            tab_av_results[policy_key][key] += np.array(domain_rtab[policy_key][key])
        #normalise
        for policy_key in tab_av_results:
            for key in tab_av_results[policy_key]:
                tab_av_results[policy_key][key] /= n_domains
                if key == 'var':
                    tab_av_results[policy_key][key] = np.square(tab_av_results[policy_key][key])
        averaged_results.append(tab_av_results)
    return averaged_results

def plotTestLogs(logfilelist,printtab,noplot,datasetname,block=True):
    """
        Extract data from given eval log files and display performance
        as a function of error rate
    """
    try:
        resultset = {}
        domains = None
        for fname in logfilelist:
            fn = open(fname,"r")
            if fn:
                lines = fn.read().splitlines()
                results = extractEvalData(lines)
                if results:
                    domains = results.keys()
                    for domain in domains:
                        resultset[domain] = {}
                        akey = results[domain].keys()[0]
                        aresult = results[domain][akey]
                        if 'policy' in aresult.keys():
                            policyname = results[domain][akey]['policy']
                            if datasetname == '':
                                i = policyname.find('-')
                                if i>=0:
                                    datasetname=policyname[:i]
                            if not policyname in resultset[domain]: resultset[domain][policyname]={}
                            for erate in results[domain].keys():
                                resultset[domain][policyname][erate] = results[domain][erate]
                        else:
                            print 'Format error in log file',fname
                            exit(0)
            else:
                print "Cannot find logfile %s" % fname
                exit(0)
        for domain in domains:
            if len(resultset[domain].keys())>0:
                (rtab,stab,ttab) = tabulateTest(resultset[domain])
                if printtab:
                    print "\n%s-%s: Performance vs Error Rate\n" % (datasetname, domain)
                    printTable('Reward', rtab)
                    printTable('Success', stab)
                    printTable('Turns', ttab)
                if not noplot:
                    plotTest(datasetname+'-'+domain,rtab,stab,block=block)
            else:
                print "No data found"
    except clog.ExceptionRaisedByLogger:
        print "Command Aborted - see Log file for error:"


@command.fetch_all('args')
def plot_command(args="",printtab=False,noplot=False,saveplot=False,datasetname=''):
    """ Call plot with a list of log files and it will print train and test curves.
        For train log files it plots performance vs num dialogs.
        For test log files it plots performance vs error rate.
        Set the printtab option to print a table of results.
        A name can be given to plot via dataset name.
    """
    trainlogs=[]
    testlogs=[]
    for fname in args:
        if fname.find('train')>=0:
            trainlogs.append(fname)
        elif fname.find('eval')>=0:
            testlogs.append(fname)
    block=True
    if testlogs: block=False
    if noplot: printtab=True    # otherwise no point!
    if trainlogs:
        plotTrainLogs(trainlogs,printtab,noplot,saveplot,datasetname,block)
    if testlogs:
        plotTestLogs(testlogs,printtab,noplot,saveplot,datasetname)


def chat_command(configfile, seed=None, trainerrorrate=None, trainsourceiteration=None):
        """ Run the texthub according to the supplied configfile.
        """
        try:
            configId = getConfigId(configfile)
            initialise(configId, configfile, seed, "chat", trainerrorrate=trainerrorrate,
                       trainsourceiteration=trainsourceiteration)
            for dom in domains:
                setupPolicy(dom, configId, gtrainerrorrate,
                            gtrainsourceiteration, gtrainsourceiteration)
                # make sure that learning is switched off
                Settings.config.set("policy_" + dom, "learning", 'False')
                # if gp, make sure to reset scale to 1 for evaluation
                if policytype == "gp":
                    Settings.config.set("gpsarsa_" + dom, "scale", "1")
            mess = "*** Chatting with policies %s: ***" % str(domains)
            if tracedialog > 0: print mess
            logger.dial(mess)

            # create text hub and run it
            hub = Texthub.ConsoleHub()
            hub.run()
            logger.dial("*** Chat complete")
            # Save a copy of config file
            confsavefile = conf_dir + configId + ".chat.cfg"
            cf = open(confsavefile, 'w')
            Settings.config.write(cf)
        except clog.ExceptionRaisedByLogger:
            print "Command Aborted - see Log file for error:",logfile
            exit(0)
        except KeyboardInterrupt:
            print "\nCommand Aborted from Keyboard"


run()
