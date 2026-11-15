// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const t=threshold.value;
threshold_line.location=t;
let tp=0,fp=0,fn=0,tn=0;
for(let k=0;k<metrics.data.score.length;k++) {
 const positive=metrics.data.label[k]===1, keep=metrics.data.score[k]>=t;
 if(positive) {if(keep)tp++;else fn++;} else {if(keep)fp++;else tn++;}
}
const tpr=tp+fn ? tp/(tp+fn) : NaN, fpr=fp+tn ? fp/(fp+tn) : NaN;
const real=prevalence.value*tpr, bogus=(1-prevalence.value)*fpr;
queue.text=`Complete evaluation: TP ${tp}, FP ${fp}, FN ${fn}, TN ${tn}. `+
 `Recall ${tpr.toFixed(4)}, FPR ${fpr.toFixed(4)}. `+
 `Expected queue per 10,000: ${(10000*(real+bogus)).toFixed(1)}; purity ${real+bogus ? (real/(real+bogus)).toFixed(4) : 'undefined'}. `+
 `Prevalence projection holds class-conditional mixtures fixed. Sliders explore saved scores.`;
let i=Math.min(index.value, bank.data.score.length-1), found=-1;
for(let k=0;k<bank.data.score.length;k++) {
 const j=(i+k)%bank.data.score.length;
 const keep=bank.data.score[j]>=t, positive=bank.data.label[j]===1;
 const outcome=(positive===keep ? 'T':'F')+(keep?'P':'N');
 if((family.value==='all'||family.value===bank.data.family[j]) && (category.value==='all'||category.value===outcome)) {found=j;break;}
}
if(found<0) {
 metadata.text='No gallery candidates match these filters. Complete metrics remain above.';
 for(const source of sources) {
  source.data={image:[],flat:[],raw_flat:[],x:[],y:[],dw:[],dh:[]};
 }
 for(const source of label_sources) {
  source.data={x:[],y:[],x_offset:[],y_offset:[],text:[]};
 }
 selection.data={x:[],y:[],width:[],height:[]};
 exact.text='No selected candidate.';
 return;
}
i=found;
if(index.value!==found)index.value=found;
metadata.text=`Candidate ${bank.data.identifier[i]}, gallery ${i+1}/${bank.data.score.length}; `+
 `family ${bank.data.family[i]}; label ${bank.data.label[i]}; score ${bank.data.score[i]}; measured SNR ${bank.data.snr[i]}.`;
const keys=['search','template','difference'];
for(let p=0;p<3;p++) {
 const values=Array.from(bank.data[keys[p]][i]);
 const sorted=values.slice().sort((a,b)=>a-b), median=sorted[Math.floor(sorted.length/2)];
 const deviations=values.map(v=>Math.abs(v-median)).sort((a,b)=>a-b);
 let sigma=1.4826*deviations[Math.floor(deviations.length/2)];
 if(!(sigma>0)) {
  const mean=values.reduce((a,b)=>a+b,0)/values.length;
  sigma=Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/values.length)||1;
 }
 const quantile=(q)=>{
  const position=q*(sorted.length-1),i=Math.floor(position),f=position-i;
  return sorted[i]+f*(sorted[Math.min(i+1,sorted.length-1)]-sorted[i]);
 };
 let low=quantile(.01), high=quantile(.99);
 if(p===2) {high=Math.max(Math.abs(low),Math.abs(high));low=-high;}
 const convert=(v)=>transform.value==='raw'?v:transform.value==='robust_z'?(v-median)/sigma:(v-(median-3*sigma))/(6*sigma);
 const display=values.map(convert);
 const matrix=Bokeh.require("core/util/ndarray").ndarray(
  new Float64Array(display), {dtype:"float64", shape:[size,size]});
 sources[p].data={image:[matrix],flat:[display],raw_flat:[values],robust_center:[median],robust_sigma:[sigma],display_center:[convert(median)],display_sigma:[transform.value==='raw'?sigma:transform.value==='robust_z'?1:1/6],transform:[transform.value],x:[0],y:[0],dw:[size],dh:[size]};
 mappers[p].low=convert(low);mappers[p].high=convert(high);
 sources[p].change.emit();
 label_sources[p].data={x:[],y:[],x_offset:[],y_offset:[],text:[]};
}
selection.data={x:[],y:[],width:[],height:[]};
exact.text='Tap any image for exact raw pixel values.';
