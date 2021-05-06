Search.setIndex({docnames:["api","api/pydra.engine","api/pydra.engine.audit","api/pydra.engine.core","api/pydra.engine.graph","api/pydra.engine.helpers","api/pydra.engine.helpers_file","api/pydra.engine.helpers_state","api/pydra.engine.specs","api/pydra.engine.state","api/pydra.engine.submitter","api/pydra.engine.task","api/pydra.engine.workers","api/pydra.mark","api/pydra.mark.functions","api/pydra.utils","api/pydra.utils.messenger","api/pydra.utils.profiler","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,sphinx:56},filenames:["api.rst","api/pydra.engine.rst","api/pydra.engine.audit.rst","api/pydra.engine.core.rst","api/pydra.engine.graph.rst","api/pydra.engine.helpers.rst","api/pydra.engine.helpers_file.rst","api/pydra.engine.helpers_state.rst","api/pydra.engine.specs.rst","api/pydra.engine.state.rst","api/pydra.engine.submitter.rst","api/pydra.engine.task.rst","api/pydra.engine.workers.rst","api/pydra.mark.rst","api/pydra.mark.functions.rst","api/pydra.utils.rst","api/pydra.utils.messenger.rst","api/pydra.utils.profiler.rst","index.rst"],objects:{"":{pydra:[0,0,0,"-"]},"pydra.engine":{AuditFlag:[1,2,1,""],DockerTask:[1,2,1,""],ShellCommandTask:[1,2,1,""],Submitter:[1,2,1,""],Workflow:[1,2,1,""],audit:[2,0,0,"-"],core:[3,0,0,"-"],graph:[4,0,0,"-"],helpers:[5,0,0,"-"],helpers_file:[6,0,0,"-"],helpers_state:[7,0,0,"-"],specs:[8,0,0,"-"],state:[9,0,0,"-"],submitter:[10,0,0,"-"],task:[11,0,0,"-"],workers:[12,0,0,"-"]},"pydra.engine.AuditFlag":{ALL:[1,3,1,""],NONE:[1,3,1,""],PROV:[1,3,1,""],RESOURCE:[1,3,1,""]},"pydra.engine.DockerTask":{container_args:[1,4,1,""],init:[1,3,1,""]},"pydra.engine.ShellCommandTask":{cmdline:[1,4,1,""],command_args:[1,4,1,""]},"pydra.engine.Submitter":{close:[1,4,1,""],submit:[1,4,1,""],submit_workflow:[1,4,1,""]},"pydra.engine.Workflow":{add:[1,4,1,""],create_connections:[1,4,1,""],done_all_tasks:[1,4,1,""],graph_sorted:[1,4,1,""],nodes:[1,4,1,""],set_output:[1,4,1,""]},"pydra.engine.audit":{Audit:[2,2,1,""]},"pydra.engine.audit.Audit":{audit_check:[2,4,1,""],audit_message:[2,4,1,""],finalize_audit:[2,4,1,""],monitor:[2,4,1,""],start_audit:[2,4,1,""]},"pydra.engine.core":{TaskBase:[3,2,1,""],Workflow:[3,2,1,""],is_lazy:[3,1,1,""],is_task:[3,1,1,""],is_workflow:[3,1,1,""]},"pydra.engine.core.TaskBase":{audit_flags:[3,3,1,""],cache_dir:[3,4,1,""],cache_locations:[3,4,1,""],can_resume:[3,4,1,""],checksum:[3,4,1,""],checksum_states:[3,4,1,""],combine:[3,4,1,""],done:[3,4,1,""],get_input_el:[3,4,1,""],help:[3,4,1,""],output_dir:[3,4,1,""],output_names:[3,4,1,""],result:[3,4,1,""],set_state:[3,4,1,""],split:[3,4,1,""],to_job:[3,4,1,""],version:[3,4,1,""]},"pydra.engine.core.Workflow":{add:[3,4,1,""],create_connections:[3,4,1,""],done_all_tasks:[3,4,1,""],graph_sorted:[3,4,1,""],nodes:[3,4,1,""],set_output:[3,4,1,""]},"pydra.engine.graph":{DiGraph:[4,2,1,""]},"pydra.engine.graph.DiGraph":{add_edges:[4,4,1,""],add_nodes:[4,4,1,""],calculate_max_paths:[4,4,1,""],copy:[4,4,1,""],edges:[4,4,1,""],edges_names:[4,4,1,""],nodes:[4,4,1,""],nodes_names_map:[4,4,1,""],remove_nodes:[4,4,1,""],remove_nodes_connections:[4,4,1,""],sorted_nodes:[4,4,1,""],sorted_nodes_names:[4,4,1,""],sorting:[4,4,1,""]},"pydra.engine.helpers":{copyfile_workflow:[5,1,1,""],create_checksum:[5,1,1,""],create_pyscript:[5,1,1,""],ensure_list:[5,1,1,""],execute:[5,1,1,""],gather_runtime_info:[5,1,1,""],get_open_loop:[5,1,1,""],hash_function:[5,1,1,""],hash_value:[5,1,1,""],load_result:[5,1,1,""],make_klass:[5,1,1,""],output_from_inputfields:[5,1,1,""],output_names_from_inputfields:[5,1,1,""],print_help:[5,1,1,""],read_and_display:[5,1,1,""],read_and_display_async:[5,1,1,""],read_stream_and_display:[5,1,1,""],record_error:[5,1,1,""],save:[5,1,1,""],task_hash:[5,1,1,""]},"pydra.engine.helpers_file":{copyfile:[6,1,1,""],copyfile_input:[6,1,1,""],copyfiles:[6,1,1,""],ensure_list:[6,1,1,""],fname_presuffix:[6,1,1,""],get_related_files:[6,1,1,""],hash_file:[6,1,1,""],is_container:[6,1,1,""],is_existing_file:[6,1,1,""],is_local_file:[6,1,1,""],on_cifs:[6,1,1,""],related_filetype_sets:[6,5,1,""],split_filename:[6,1,1,""],template_update:[6,1,1,""]},"pydra.engine.helpers_state":{add_name_combiner:[7,1,1,""],add_name_splitter:[7,1,1,""],combine_final_groups:[7,1,1,""],connect_splitters:[7,1,1,""],converter_groups_to_input:[7,1,1,""],flatten:[7,1,1,""],input_shape:[7,1,1,""],inputs_types_to_dict:[7,1,1,""],iter_splits:[7,1,1,""],map_splits:[7,1,1,""],remove_inp_from_splitter_rpn:[7,1,1,""],rpn2splitter:[7,1,1,""],splits:[7,1,1,""],splits_groups:[7,1,1,""],splitter2rpn:[7,1,1,""]},"pydra.engine.specs":{BaseSpec:[8,2,1,""],ContainerSpec:[8,2,1,""],Directory:[8,2,1,""],DockerSpec:[8,2,1,""],File:[8,2,1,""],LazyField:[8,2,1,""],Result:[8,2,1,""],Runtime:[8,2,1,""],RuntimeSpec:[8,2,1,""],ShellOutSpec:[8,2,1,""],ShellSpec:[8,2,1,""],SingularitySpec:[8,2,1,""],SpecInfo:[8,2,1,""],TaskHook:[8,2,1,""],attr_fields:[8,1,1,""],donothing:[8,1,1,""],path_to_string:[8,1,1,""]},"pydra.engine.specs.BaseSpec":{check_fields_input_spec:[8,4,1,""],check_metadata:[8,4,1,""],collect_additional_outputs:[8,4,1,""],copyfile_input:[8,4,1,""],hash:[8,4,1,""],retrieve_values:[8,4,1,""],template_update:[8,4,1,""]},"pydra.engine.specs.ContainerSpec":{bindings:[8,3,1,""],container:[8,3,1,""],container_xargs:[8,3,1,""],image:[8,3,1,""]},"pydra.engine.specs.DockerSpec":{container:[8,3,1,""]},"pydra.engine.specs.LazyField":{get_value:[8,4,1,""]},"pydra.engine.specs.Result":{errored:[8,3,1,""],output:[8,3,1,""],runtime:[8,3,1,""]},"pydra.engine.specs.Runtime":{cpu_peak_percent:[8,3,1,""],rss_peak_gb:[8,3,1,""],vms_peak_gb:[8,3,1,""]},"pydra.engine.specs.RuntimeSpec":{container:[8,3,1,""],network:[8,3,1,""],outdir:[8,3,1,""]},"pydra.engine.specs.ShellOutSpec":{collect_additional_outputs:[8,4,1,""],return_code:[8,3,1,""],stderr:[8,3,1,""],stdout:[8,3,1,""]},"pydra.engine.specs.ShellSpec":{args:[8,3,1,""],check_fields_input_spec:[8,4,1,""],check_metadata:[8,4,1,""],executable:[8,3,1,""],retrieve_values:[8,4,1,""]},"pydra.engine.specs.SingularitySpec":{container:[8,3,1,""]},"pydra.engine.specs.SpecInfo":{bases:[8,3,1,""],fields:[8,3,1,""],name:[8,3,1,""]},"pydra.engine.specs.TaskHook":{post_run:[8,3,1,""],post_run_task:[8,3,1,""],pre_run:[8,3,1,""],pre_run_task:[8,3,1,""],reset:[8,4,1,""]},"pydra.engine.state":{State:[9,2,1,""]},"pydra.engine.state.State":{combiner:[9,4,1,"id0"],connect_groups:[9,4,1,""],connect_splitters:[9,4,1,""],final_combined_ind_mapping:[9,3,1,""],group_for_inputs:[9,3,1,""],group_for_inputs_final:[9,3,1,""],groups_stack_final:[9,3,1,""],inner_inputs:[9,3,1,""],inputs_ind:[9,3,1,""],merge_previous_states:[9,4,1,""],name:[9,3,1,""],other_states:[9,3,1,""],prepare_inputs:[9,4,1,""],prepare_states:[9,4,1,""],prepare_states_combined_ind:[9,4,1,""],prepare_states_ind:[9,4,1,""],prepare_states_val:[9,4,1,""],push_new_states:[9,4,1,""],set_input_groups:[9,4,1,""],set_splitter_final:[9,4,1,""],splitter:[9,4,1,"id1"],splitter_final:[9,3,1,""],splitter_rpn:[9,3,1,""],splitter_rpn_compact:[9,3,1,""],states_ind:[9,3,1,""],states_val:[9,3,1,""]},"pydra.engine.submitter":{Submitter:[10,2,1,""],get_runnable_tasks:[10,1,1,""],is_runnable:[10,1,1,""]},"pydra.engine.submitter.Submitter":{close:[10,4,1,""],submit:[10,4,1,""],submit_workflow:[10,4,1,""]},"pydra.engine.task":{ContainerTask:[11,2,1,""],DockerTask:[11,2,1,""],FunctionTask:[11,2,1,""],ShellCommandTask:[11,2,1,""],SingularityTask:[11,2,1,""]},"pydra.engine.task.ContainerTask":{bind_paths:[11,4,1,""],binds:[11,4,1,""],container_check:[11,4,1,""]},"pydra.engine.task.DockerTask":{container_args:[11,4,1,""],init:[11,3,1,""]},"pydra.engine.task.ShellCommandTask":{cmdline:[11,4,1,""],command_args:[11,4,1,""]},"pydra.engine.task.SingularityTask":{container_args:[11,4,1,""],init:[11,3,1,""]},"pydra.engine.workers":{ConcurrentFuturesWorker:[12,2,1,""],DistributedWorker:[12,2,1,""],SerialPool:[12,2,1,""],SerialWorker:[12,2,1,""],SlurmWorker:[12,2,1,""],Worker:[12,2,1,""]},"pydra.engine.workers.ConcurrentFuturesWorker":{close:[12,4,1,""],exec_as_coro:[12,4,1,""],run_el:[12,4,1,""]},"pydra.engine.workers.DistributedWorker":{fetch_finished:[12,4,1,""],max_jobs:[12,3,1,""]},"pydra.engine.workers.SerialPool":{done:[12,4,1,""],result:[12,4,1,""],submit:[12,4,1,""]},"pydra.engine.workers.SerialWorker":{close:[12,4,1,""],run_el:[12,4,1,""]},"pydra.engine.workers.SlurmWorker":{run_el:[12,4,1,""]},"pydra.engine.workers.Worker":{close:[12,4,1,""],fetch_finished:[12,4,1,""],run_el:[12,4,1,""]},"pydra.mark":{functions:[14,0,0,"-"]},"pydra.mark.functions":{annotate:[14,1,1,""],task:[14,1,1,""]},"pydra.utils":{messenger:[16,0,0,"-"],profiler:[17,0,0,"-"]},"pydra.utils.messenger":{AuditFlag:[16,2,1,""],FileMessenger:[16,2,1,""],Messenger:[16,2,1,""],PrintMessenger:[16,2,1,""],RemoteRESTMessenger:[16,2,1,""],RuntimeHooks:[16,2,1,""],collect_messages:[16,1,1,""],gen_uuid:[16,1,1,""],make_message:[16,1,1,""],now:[16,1,1,""],send_message:[16,1,1,""]},"pydra.utils.messenger.AuditFlag":{ALL:[16,3,1,""],NONE:[16,3,1,""],PROV:[16,3,1,""],RESOURCE:[16,3,1,""]},"pydra.utils.messenger.FileMessenger":{send:[16,4,1,""]},"pydra.utils.messenger.Messenger":{send:[16,4,1,""]},"pydra.utils.messenger.PrintMessenger":{send:[16,4,1,""]},"pydra.utils.messenger.RemoteRESTMessenger":{send:[16,4,1,""]},"pydra.utils.messenger.RuntimeHooks":{resource_monitor_post_stop:[16,3,1,""],resource_monitor_pre_start:[16,3,1,""],task_execute_post_exit:[16,3,1,""],task_execute_pre_entry:[16,3,1,""],task_run_entry:[16,3,1,""],task_run_exit:[16,3,1,""]},"pydra.utils.profiler":{ResourceMonitor:[17,2,1,""],get_max_resources_used:[17,1,1,""],get_system_total_memory_gb:[17,1,1,""],log_nodes_cb:[17,1,1,""]},"pydra.utils.profiler.ResourceMonitor":{fname:[17,4,1,""],run:[17,4,1,""],stop:[17,4,1,""]},pydra:{check_latest_version:[0,1,1,""],engine:[1,0,0,"-"],mark:[13,0,0,"-"],utils:[15,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","attribute","Python attribute"],"4":["py","method","Python method"],"5":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:attribute","4":"py:method","5":"py:data"},terms:{"8192":6,"abstract":16,"case":[2,5],"class":[0,1,2,3,4,5,6,7,8,9,10,11,12,16,17],"default":[1,6,8,10],"enum":[1,16],"final":[3,7,9,12],"float":[8,14,17],"function":[0,6,7,8,13,17],"import":[6,9,14],"int":[8,14,17],"new":[4,5,12,17],"return":[1,3,4,5,6,7,8,10,11,12,14,17],"true":[1,5,6,7,10,16],That:[1,3],The:[0,1,3,5,7,8,16],Uses:[5,9,11],_na:9,_node_wip:4,abc:5,abl:6,about:9,absolut:6,accept:[3,17],access:6,actual:[1,11],add:[1,3,4,7,9],add_edg:4,add_name_combin:7,add_name_splitt:7,add_nod:4,added:[1,3,4],adding:7,addit:[7,8,9],address:9,administr:5,afil:6,afni:6,after:[6,9],aggreg:[1,10],algorithm:7,all:[1,3,4,6,7,8,9,10,16],allow:16,alreadi:[5,6],also:[4,6,8,9],analyz:6,ani:[1,3,4,8,10,17],annot:14,api:[12,18],append:[6,16],appli:14,applic:18,arg:[1,8,11],argument:[1,8,11],arriv:5,associ:9,assum:6,async:[1,5,10,12],asyncio:[5,12],attr:12,attr_field:8,attr_typ:8,audit:[0,1,3,11,16],audit_check:2,audit_flag:[1,2,3,11],audit_messag:2,auditflag:[1,3,11,16],avail:[1,3,8,9],await:[1,10,12],axes:[7,9],axi:7,backend:[1,10],bad:0,base:[1,2,3,4,6,8,9,10,11,12,16,17],basespec:[1,3,8,11],basic:[3,8],been:3,befor:[1,10],behavior:6,being:[0,17],between:[4,9],bind:[8,11],bind_path:11,binds_path:11,bool:[0,1,3,5,6,7,8,10,16],both:3,bound:[8,11],brik:6,build:16,built:6,cach:[3,5],cache_dir:[1,3,11],cache_loc:[1,3,5,11],calcul:[3,4,5,9],calculate_max_path:4,call:17,callabl:[8,11],callback:[11,17],can:[1,3,6,9],can_hardlink:6,can_resum:3,can_symlink:6,captur:5,certain:17,chang:7,charact:6,check:[0,1,3,6,8,10],check_fields_input_spec:8,check_latest_vers:0,check_metadata:8,checkpoint:3,checksum:[3,5],checksum_st:3,chunk_len:6,cif:6,cli:[1,11],close:[1,5,10,12],cmd:5,cmdline:[1,11],code:8,collect:[3,5,8],collect_additional_output:8,collect_messag:16,collected_path:16,combin:[3,5,7,9],combine_final_group:7,come:9,command:[1,5,6,8,11],command_arg:[1,11],compact:[9,16],complet:[1,10],composit:[1,3],comput:[1,3,6,8],concurr:[6,12],concurrentfutureswork:12,conda:11,config:11,connect:[1,3,4,9],connect_group:9,connect_splitt:[7,9],consid:17,consumpt:8,contain:[1,4,5,6,8,9,11],container:[1,8,11],container_arg:[1,11],container_check:11,container_info:[1,11],container_typ:11,container_xarg:8,containerspec:8,containertask:[1,11],context:16,convers:7,convert:[7,8],converter_groups_to_input:7,copi:[4,5,8,11],copy_related_fil:6,copyfil:6,copyfile_input:[6,8],copyfile_workflow:5,core:[0,1,11,17],coroutin:[1,5,10,12],could:9,cpath:11,cpu:8,cpu_peak_perc:8,creat:[2,4,5,6,9],create_checksum:5,create_connect:[1,3],create_new:6,create_pyscript:5,crypto:6,cur_depth:7,current:[4,5,9],cwl:8,data:[4,5,6,8],dataclass:8,decor:14,def:14,defin:6,depend:9,design:8,dest:6,destin:6,determin:[2,9],develop:2,dict:[3,6,9,11,16],dictionari:[4,7,9,17],diff:6,differ:[5,6],digraph:4,direct:4,directori:[2,3,5,6,8,11],disabl:6,discuss:5,displai:5,distribut:[1,10,12],distributedwork:12,dmtcp:11,docker:[1,6,8,11],dockerrequir:8,dockerspec:8,dockertask:[1,11],doe:[4,17],doesn:[1,3],done:[1,3,12],done_all_task:[1,3],donoth:8,driver:6,due:7,duplic:4,dure:17,each:[7,9],ecosystem:0,edg:4,edges_nam:4,either:[4,6],element:[1,3,9,11],elements_to_remove_comb:9,elemntari:3,enabl:2,end:[2,7,17],endpoint:16,engin:[0,17],ensure_list:[5,6],entrypoint:[1,10],env:11,environ:[5,11],envvarrequir:8,eof:5,error:[1,5,6,8,10,17],error_path:5,especi:9,evalu:9,event:5,eventloop:5,everi:[7,9],exampl:[5,6,14],exec_as_coro:12,execut:[1,5,8,10,11,12,17],executor:12,exist:[1,3],exit:8,expos:6,ext:6,extend:[1,11],extens:6,extern:11,extract:5,fact:[1,3],failur:6,fals:[0,1,3,5,6,7,8,10,11,12,17],far:17,featur:6,fetch_finish:12,field:[3,5,7,8,9],file:[3,5,8,11,16,17],filelist:6,filemesseng:16,filenam:[6,17],filenotfounderror:6,filesystem:[3,6,11],filetyp:6,final_combined_ind_map:9,finalize_audit:2,find:6,finish:12,first:0,flag:[1,2,3,16],flatten:7,fname:[5,6,17],fname_presuffix:6,folder:8,foo:6,form:0,format:16,found:6,fragment:11,framework:5,french:6,frequenc:17,from:[1,3,4,5,6,7,8,9,10,11],fulfil:8,full:[6,9],func:[11,14],functiontask:[11,14],futur:[1,10,12],gather:16,gather_runtime_info:5,gen_uuid:16,gener:[3,5,7,8,16],get:[1,3,4,5,7,8,9,11,12,16,17],get_input_el:3,get_max_resources_us:17,get_open_loop:5,get_related_fil:6,get_runnable_task:10,get_system_total_memory_gb:17,get_valu:8,given:[5,6,8,17],graph:[0,1,3,10],graph_sort:[1,3],greater:6,group:[7,9],group_for_input:[7,9],group_for_inputs_fin:9,groups_stack:7,groups_stack_fin:9,handl:[2,10],hard:6,hardlink:6,has:[1,3,6,11],hash:[5,6,8,11],hash_fil:6,hash_funct:5,hash_valu:5,have:[6,7,9],hdr:6,head:6,help:3,helper:[0,1],helpers_fil:[0,1],helpers_st:[0,1],hide_displai:5,high:17,histori:4,hlpst:9,home:6,hook:[8,16],host:6,identifi:[5,16],imag:8,img:6,imit:12,implement:[6,8,11],impos:7,in1:7,includ:[6,9],include_this_fil:6,inconsist:6,ind:[3,11],index:18,indic:9,info:17,inform:[2,5,9],inherit:[3,8],init:[1,11],initi:[1,10],initialworkdirrequir:8,inlinejavascriptrequir:8,inlinescriptrequir:8,inner:9,inner_input:[7,9],inp:9,input:[3,5,6,7,8,9],input_shap:7,input_spec:[1,3,8,11],inputs_ind:9,inputs_to_remov:7,inputs_types_to_dict:7,insert:4,insid:2,instanc:[1,3,10],instead:6,integ:17,intenum:16,interfac:[3,5,9,12,18],intern:[11,12,17],interpret:6,interv:17,invok:8,is_contain:6,is_existing_fil:6,is_lazi:3,is_local_fil:6,is_runn:10,is_task:3,is_workflow:3,isol:11,item:6,iter:7,iter_split:7,its:5,job:12,join:0,json:[11,17],keep:[2,6,8,9,17],kei:7,kwarg:[1,3,8,10,11,12,16],latest:0,lazi:[3,8],lazyfield:8,ld_op:16,lead:6,left:[7,9],length:[6,7],librari:18,like:3,limit:12,line:[1,5,8,11],linearli:12,link:[4,6,7],list:[1,3,4,5,6,7,8,9,11],load:5,load_result:5,local:11,locat:3,log:[16,17],log_nodes_cb:17,logdir:17,logger:17,look:[3,5],loop:[1,5,10,12],lpath:11,mai:6,make_klass:5,make_messag:16,mandatori:8,manipul:6,map:[0,4,9],map_copyfil:6,map_split:7,mark:[0,4],mat:6,max_depth:7,max_job:12,maximum:[4,12],mean:[1,3],medatada:8,mem_mb:17,memori:[8,17],merg:9,merge_previous_st:9,messag:[2,16],message_path:16,messeng:[0,1,2,3,11,15],messenger_arg:[1,2,3,11],metadata:[5,8],meth:11,method:[4,6],minshal:6,mode:11,modifi:6,modul:[0,1,13,15,18],monitor:[1,2,11,16,17],more:3,mostli:7,mount:[6,8,11],mutat:7,n_proc:12,name:[1,3,4,5,6,7,8,9,11],need:[4,8,9],network:8,neurodock:11,neuroimag:6,new_edg:4,new_nod:4,newfil:6,newpath:6,niceman:11,nidm:16,nifti:6,nii:6,nipyp:[0,6,17],node:[1,3,4,7,8,9,11,17],node_st:3,nodes_names_map:4,none:[1,2,3,4,5,6,7,8,9,10,11,12,16,17],notat:[7,9],noth:8,now:16,num_thread:17,number:[7,9,12],obj:[3,5,10,16],object:[1,2,3,4,5,6,8,9,10,12,16],odir:2,on_cif:6,one:[3,7,12],onli:[1,3,6,9,11,16],open:16,openssl_sha256:6,oper:[0,6,9],opt:11,option:[1,3,8,11],order:[5,9],origin:[6,11],originalfil:6,other:6,other_st:[7,9],otherwis:[6,17],outcom:3,outdir:8,outer:9,output:[1,2,3,5,6,8,9,16],output_dir:[3,6,8],output_file_templ:5,output_from_inputfield:5,output_nam:3,output_names_from_inputfield:5,output_spec:[1,3,5,8,11],over:[3,6,9],overwrit:3,packag:0,page:18,pair:[4,6,7],parallel:12,paramet:[0,1,2,3,4,5,6,7,10,12,16,17],parameter:3,parametr:3,pars:[8,10],part:[6,9],partial:9,particular:[1,2,3,8],path:[3,4,5,6,8],path_to_str:8,pathlib:[5,8],pathlik:[2,5,8],peak:8,pend:12,perform:[7,17],physic:8,pickl:5,pid:17,pipelin:17,plugin:[1,10],point:[8,11,16],polish:7,poll_delai:12,pool:12,port:6,posix:6,post:6,post_run:8,post_run_task:8,pre:6,pre_run:8,pre_run_task:8,preced:6,predecessor:4,prefix:6,prefoopost:6,prepar:9,prepare_input:9,prepare_st:9,prepare_states_combined_ind:9,prepare_states_ind:9,prepare_states_v:9,prepend:6,prescrib:7,present:6,presort:4,previou:[3,9],previous:[1,4,10],print:[3,5,16],print_help:5,printmesseng:16,prioriti:5,process:[3,5,7,8,9,11,16,17],profil:[0,15],programm:18,promis:8,promot:14,properti:[1,3,4,8,9,11,17],prov:[1,16],proven:[1,2,11,16],provid:[7,8],prune:4,pth:6,push_new_st:9,py2:6,pydra:0,pyfunc:17,pyscript:5,python:[11,12],rais:0,raise_except:0,raise_notfound:6,ram:[8,17],read:[5,6,11],read_and_displai:5,read_and_display_async:5,read_stream_and_displai:5,readi:9,recent:6,record:[2,17],record_error:5,recurr:7,recurs:5,redirect:16,reduc:9,refer:4,refin:8,regard:8,regular:6,relat:6,related_filetype_set:6,relev:9,remot:[11,16],remoterestmesseng:16,remov:[1,4,6,7,9,10],remove_inp_from_splitter_rpn:7,remove_nod:4,remove_node_connect:4,remove_nodes_connect:4,replac:[3,6],report:[1,3],repres:[8,9],represent:[1,3],requir:[3,7,8],rerun:[1,5,10,12],reset:8,resourc:[1,2,16,17],resource_monitor_post_stop:16,resource_monitor_pre_start:16,resourcemonitor:17,resourcerequir:8,rest:16,restart:3,restor:5,result:[1,2,3,5,6,8,9,12],resum:11,retriev:3,retrieve_valu:8,return_cod:8,returnhelp:3,revers:7,rewrit:0,right:[7,9],rpn2splitter:7,rpn:[7,9],rss_peak_gb:8,run:[1,3,4,5,6,8,9,10,12,17],run_el:12,runnabl:[1,4,10,12],runtim:[5,8],runtimeerror:0,runtimehook:16,runtimespec:8,same:[4,6,9],save:5,sbatch_arg:12,scalar:9,schemadefrequir:8,script:5,script_path:5,search:18,see:4,self:[4,9],send:[1,2,10,12,16],send_messag:16,sent:2,serialpool:12,serialwork:12,server:11,set:[1,3,5,6,8,10,12,17],set_input_group:9,set_output:[1,3],set_splitter_fin:9,set_stat:3,shape:7,share:6,shell:[1,8,11],shellcommandrequir:8,shellcommandtask:[1,11],shelloutspec:8,shellspec:8,shelltask:8,should:[6,9],simpl:[4,12],singl:9,singular:[8,11],singularityspec:8,singularitytask:11,slurm:12,slurmwork:12,softwarerequir:8,sort:[1,3,4],sorted_nod:4,sorted_nodes_nam:4,sourc:3,spec:[0,1,3,5,6,11],specif:[1,3,8,9,11,17],specifi:[6,7,9,11],specinfo:[8,11],split:[3,6,7,9],split_filenam:6,split_it:7,splits_group:7,splitter2rpn:7,splitter:[3,7,9],splitter_fin:9,splitter_rpn:[7,9],splitter_rpn_compact:9,spm:6,squar:14,stack:9,stackoverflow:5,standalon:5,standard:[5,8,16],start:[2,4,17],start_audit:2,state:[0,1,3,7,10,11,16],state_field:7,state_index:[3,8],states_ind:9,states_v:9,statist:17,statu:17,stderr:8,stdout:8,step:3,stop:17,store:3,str:[1,3,5,6,8,9],stream:5,string:[5,6,8,17],strip:5,structur:[1,3,4,8],subject:6,submiss:[1,10,12],submit:[1,10,11,12],submit_workflow:[1,10],submitt:[0,1],submodul:0,subpackag:18,suffix:6,support:[4,5,6],symbol:6,symlink:6,system:[6,11,12,17],take:6,task:[0,1,2,3,4,5,8,9,10,12,14],task_execute_post_exit:16,task_execute_pre_entri:16,task_hash:5,task_path:5,task_run_entri:16,task_run_exit:16,taskbas:[1,3,5,10,11],taskhook:8,templat:[6,8],template_upd:[6,8],text:6,than:6,thei:[4,5],them:[1,4,5,10],thi:[3,4,5,6,8,12,17],thread:17,through:6,time:[7,8],timestamp:16,tmp:6,to_job:3,todo:[1,2,3,5,7],togeth:6,total:17,track:[1,2,8,9,11,16,17],translat:7,truncat:16,tupl:[5,6,8,9],tuple2list:5,type:[1,3,5,6,7,8,9,10,12,17],under:6,union:[1,3,8],uniqu:[3,5,16],unless:5,unlink:6,until:[2,5,12],unwrap:9,updat:[6,8,14],use_ext:6,use_hardlink:6,used:[0,3,5,6,7,9,14],user:7,using:[6,9,12],util:[0,1,2,3,11],val:7,valu:[1,3,5,7,8,9,16,17],variabl:11,version:[0,3],virtual:8,visit:5,vms_peak_gb:8,wait:[1,10],watermark:17,were:6,wf_path:5,what:3,whatev:5,when:[4,6,7,8,9,16],where:[3,5],whether:[2,3,6,12],which:[3,4,5,6,9],window:6,within:10,without:[4,6],work:[5,11],worker:[0,1,10],workflow:[0,1,3,4,5,10,14],wrap:[1,11],wrapper:12,write:[1,3,5,11],written:[3,6],xor:8},titles:["Library API (application programmer interface)","pydra.engine package","pydra.engine.audit module","pydra.engine.core module","pydra.engine.graph module","pydra.engine.helpers module","pydra.engine.helpers_file module","pydra.engine.helpers_state module","pydra.engine.specs module","pydra.engine.state module","pydra.engine.submitter module","pydra.engine.task module","pydra.engine.workers module","pydra.mark package","pydra.mark.functions module","pydra.utils package","pydra.utils.messenger module","pydra.utils.profiler module","Welcome to Pydra: A simple dataflow engine with scalable semantics\u2019s documentation!"],titleterms:{"function":14,"new":6,api:0,applic:0,audit:2,content:18,copi:6,core:3,dataflow:18,document:18,engin:[1,2,3,4,5,6,7,8,9,10,11,12,18],exist:6,file:6,graph:4,helper:5,helpers_fil:6,helpers_st:7,indic:18,interfac:0,librari:0,mark:[13,14],messeng:16,modul:[2,3,4,5,6,7,8,9,10,11,12,14,16,17],note:11,option:6,packag:[1,13,15],profil:17,programm:0,pydra:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],scalabl:18,semant:18,simpl:18,spec:8,state:9,submitt:10,submodul:[1,13,15],subpackag:0,tabl:18,task:11,util:[15,16,17],welcom:18,worker:12}})