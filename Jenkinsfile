#!groovy

@Library('katsdpjenkins') _
katsdp.killOldJobs()

katsdp.setDependencies([
    'ska-sa/katsdpdockerbase/master',
    'ska-sa/katpoint/master',
    'ska-sa/katdal/master',
    'ska-sa/katsdptelstate/master'])
katsdp.standardBuild()
katsdp.mail('sdpdev+katsdpcal@ska.ac.za')
