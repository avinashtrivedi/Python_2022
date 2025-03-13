ruCand = rspJson[0]["reportingSource"]["candidates"]
resp_id_total = []

for ruval in ruCand:
    for indx in range(len(ruval['vote']['parties'])):
        cid = ruval['vote']['parties'][indx]['candidateId'] 
        rsp_pid = ruval['vote']['parties'][indx]['id'] 
        
        for can_id,total,pid in vote_id_total:
            logger.info({'can_id':can_id,'cid':cid})
            if pid == rsp_pid and can_id == cid:
                logger.info({'can_id':can_id,'cid':cid})
                ruval['vote']['parties'][indx]['vote']['total'] = total
                resp_id_total.append((cid,ruval['vote']['parties'][indx]['vote']['total']))
                logger.info({'Updated val->':ruval['vote']['parties'][indx]['vote']['total']})
            
resp_id_total = list(set(resp_id_total))
resp_id_total.sort(key = lambda x:x[1],reverse=True)           
logger.info({'Max vote':resp_id_total[0][0]}) 