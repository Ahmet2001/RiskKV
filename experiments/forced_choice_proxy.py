"""Minimal forced-choice proxy for risk-routed heterogeneous KV memory.

This script is intentionally compact. It builds synthetic exact-recall prompts,
computes candidate answer NLL under Full KV, quant-only, and ours-core policies,
and reports accuracy plus effective KV memory ratio. The current implementation
simulates quantization by quantize/dequantize operations on FP16 tensors; it is
not a packed low-bit KV backend.
"""
import argparse, csv, os, random, time
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

FILLERS=[" The archive contains routine notes about unrelated projects."," This paragraph is ordinary context and should not require exact recall."," The report repeats administrative information and background details."]
MARKERS=["MK53-4897X","ZX17-2044Q","ALPHA-77K2","NOVA-9031B","RIVER-42M","ORBIT-118X","CYPHER-8007Z","DELTA-330Q","TITAN-991K","VECTOR-118B"]
DATES=["2031-07-18","2029-11-04","2034-02-25","2028-05-12","2032-09-30"]

def enc(tok,s): return tok.encode(s, add_special_tokens=False)
def mean(xs): return sum(xs)/max(1,len(xs))
def add(tok, ids, text, tag, tags):
    st=len(ids); ts=enc(tok,text); ids.extend(ts)
    for i in range(st,st+len(ts)): tags.setdefault(tag,set()).add(i)
def fill_to(tok,ids,tags,target,rng):
    while len(ids)<target: add(tok,ids,rng.choice(FILLERS),"bg",tags)
def expand(pos,n,w):
    out=set()
    for p in pos: out.update(range(max(0,p-w), min(n,p+w+1)))
    return out

def build_case(tok,ctx,cid):
    rng=random.Random(18000+ctx*13+cid*97); ids=[]; tags={}
    gold=MARKERS[cid%len(MARKERS)]
    fill_to(tok,ids,tags,int(ctx*0.45),rng)
    add(tok,ids," Secret marker:","support",tags); add(tok,ids," "+gold,"exact",tags); add(tok,ids,".","support",tags)
    fill_to(tok,ids,tags,ctx,rng); ids=ids[:ctx]
    for k in tags: tags[k]={p for p in tags[k] if p<len(ids)}
    query=enc(tok," Question: Which marker was labeled as the secret marker? Answer:")
    choices=[enc(tok," "+gold)]+[enc(tok," "+m) for m in MARKERS if m!=gold][:4]
    return dict(prefix=ids,query=query,choices=choices,gold=gold,exact=tags.get("exact",set()))

def tiers(policy,n,exact,bits,window):
    if policy=="full": return ["full"]*n
    if policy=="quant_only": return ["quant"]*n
    keep=expand(exact,n,window)
    return ["full" if i in keep else "quant" for i in range(n)]
def mem_ratio(ts,bits): return mean([1.0 if t=="full" else bits/16.0 for t in ts])
def qtz(x,bits):
    qmax=(2**(bits-1))-1; scale=x.abs().amax(dim=-1,keepdim=True).clamp(min=1e-6)/qmax
    return torch.clamp((x/scale).round(),-qmax,qmax)*scale

def apply_past(past,ts,bits):
    if all(t=="full" for t in ts): return past
    dev=past[0][0].device; qidx=torch.tensor([i for i,t in enumerate(ts) if t=="quant"],device=dev)
    out=[]
    for k,v in past:
        kk=k.clone(); vv=v.clone()
        if qidx.numel(): kk[:,:,qidx,:]=qtz(kk[:,:,qidx,:],bits); vv[:,:,qidx,:]=qtz(vv[:,:,qidx,:],bits)
        out.append((kk,vv))
    return tuple(out)

@torch.no_grad()
def seq_nll(model,device,past,plen,query,answer,bits,ts):
    mod=apply_past(past,ts,bits); q=torch.tensor([query],device=device)
    att=torch.ones((1,plen+len(query)),device=device,dtype=torch.long)
    out=model(input_ids=q,past_key_values=mod,attention_mask=att,use_cache=True)
    cp=out.past_key_values; logits=out.logits[:,-1,:]; total=0.0; curlen=plen+len(query)
    for t in answer:
        total += float(-torch.log_softmax(logits.float(),-1)[0,t].cpu())
        inp=torch.tensor([[t]],device=device); curlen+=1
        att=torch.ones((1,curlen),device=device,dtype=torch.long)
        out=model(input_ids=inp,past_key_values=cp,attention_mask=att,use_cache=True)
        cp=out.past_key_values; logits=out.logits[:,-1,:]
    return total

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct'); ap.add_argument('--context',type=int,default=2048); ap.add_argument('--bits',type=int,default=4); ap.add_argument('--cases',type=int,default=3); ap.add_argument('--window',type=int,default=4); ap.add_argument('--out',default='outputs'); args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True); device='cuda' if torch.cuda.is_available() else 'cpu'
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,attn_implementation='sdpa',torch_dtype=torch.float16 if device=='cuda' else None).to(device).eval()
    policies=[('full',16),('quant_only',args.bits),('ours_core',args.bits)]; rows=[]
    for cid in range(args.cases):
        case=build_case(tok,args.context,cid); ids=torch.tensor([case['prefix']],device=device)
        out=model(input_ids=ids,use_cache=True); past=out.past_key_values
        for pol,bits in policies:
            ts=tiers(pol,len(case['prefix']),case['exact'],bits,args.window)
            vals=[seq_nll(model,device,past,len(case['prefix']),case['query'],ch,bits,ts) for ch in case['choices']]
            pred=min(range(len(vals)),key=lambda i: vals[i]); margin=min(vals[1:])-vals[0]
            rows.append({'case_id':cid,'policy':pol,'bits':bits,'memory_ratio':mem_ratio(ts,bits),'acc':int(pred==0),'gold_nll':vals[0],'margin':margin})
    with open(os.path.join(args.out,'raw_results.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for pol,_ in policies:
        rs=[r for r in rows if r['policy']==pol]
        print(pol, 'acc', mean([r['acc'] for r in rs]), 'mem', mean([r['memory_ratio'] for r in rs]), 'gold_nll', mean([r['gold_nll'] for r in rs]))
if __name__=='__main__': main()
