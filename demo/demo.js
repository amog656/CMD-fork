const examples = [
  {name:'DEMO CANDIDATE 01',smiles:'O=C(Nc1ccccc1)Nc1ncc(N2CCCCC2)n1',mw:'318.37',qed:'0.78',logp:'2.41',vina:'−7.2',sas:'2.8'},
  {name:'DEMO CANDIDATE 02',smiles:'COc1ccc(C(=O)Nc2ncccn2)cc1',mw:'269.26',qed:'0.71',logp:'2.06',vina:'−6.8',sas:'2.4'},
  {name:'DEMO CANDIDATE 03',smiles:'CCN1CCN(c2ncnc3ccccc23)CC1',mw:'256.31',qed:'0.83',logp:'2.18',vina:'−6.4',sas:'2.2'},
  {name:'DEMO CANDIDATE 04',smiles:'O=C(O)c1ccc(Nc2ncccn2)cc1',mw:'243.22',qed:'0.76',logp:'1.73',vina:'−6.1',sas:'2.1'},
  {name:'DEMO CANDIDATE 05',smiles:'CCOc1cc(NC(=O)C2CC2)ccc1OC',mw:'263.29',qed:'0.69',logp:'2.32',vina:'−5.9',sas:'2.6'},
  {name:'DEMO CANDIDATE 06',smiles:'N#Cc1ccc(-n2ccnc2)cc1',mw:'ಿರ',qed:'0.74',logp:'1.91',vina:'−5.7',sas:'2.3'},
  {name:'DEMO CANDIDATE 07',smiles:'O=C(NCC1CC1)c1ncc(-c2ccccc2)o1',mw:'268.31',qed:'0.72',logp:'2.26',vina:'−5.5',sas:'2.7'},
  {name:'DEMO CANDIDATE 08',smiles:'CC1(C)NCCN1c1ncc(Cl)cc1',mw:'225.68',qed:'0.66',logp:'2.11',vina:'−5.3',sas:'2.5'}
];
// This page intentionally uses fabricated examples to demonstrate the interface only.
examples[5].mw='219.24';
const $=s=>document.querySelector(s);
const count=$('#count'), state=$('#states'), exhaustiveness=$('#exhaustiveness');
let selected=0, zoom=1, rotation=0, drag=null, runTimer=null;

function updateRange(input,output){$(output).value=input.value;$(output).textContent=input.value;}
state.addEventListener('input',()=>{updateRange(state,'#state-value');renderStructure();});
exhaustiveness.addEventListener('input',()=>updateRange(exhaustiveness,'#exh-value'));
$('#complex-file').addEventListener('change',e=>{
  const file=e.target.files?.[0];
  if(file) $('#file-label').textContent=file.name+' (local demo selection)';
});

function renderCandidates(){
  const items=examples.slice(0,Number(count.value));
  $('#candidate-list').innerHTML=items.map((c,i)=>`<button class="candidate ${i===selected?'selected':''}" data-index="${i}" aria-pressed="${i===selected}"><span class="number">${String(i+1).padStart(2,'0')}</span><span><strong>${c.name}</strong><small>${c.smiles}</small></span><span class="metric"><small>MW</small>${c.mw}</span><span class="metric"><small>MEAN VINA*</small>${c.vina}</span></button>`).join('');
  $('#candidate-list').querySelectorAll('.candidate').forEach(button=>button.addEventListener('click',()=>selectCandidate(Number(button.dataset.index))));
  renderStructure();
}
function selectCandidate(index){selected=index;zoom=1;rotation=0;renderCandidates();}
function renderStructure(){
  const candidate=examples[selected];
  $('#viewer-index').textContent=`${String(selected+1).padStart(2,'0')} / ${String(Math.min(examples.length,Number(count.value))).padStart(2,'0')}`;
  $('#viewer-name').textContent=candidate.name;
  $('#viewer-smiles').textContent=candidate.smiles;
  $('#property-grid').innerHTML=[['MW (DA)',candidate.mw],['QED',candidate.qed],['SAS*',candidate.sas],['LOGP',candidate.logp],['MEAN VINA*',candidate.vina],['STATES',state.value]].map(([label,value])=>`<div class="property"><small>${label}</small><strong>${value}</strong></div>`).join('');
  // Decorative 2D atom-and-bond drawing; it does not encode the SMILES above.
  const points=[[180,55],[228,82],[228,137],[180,164],[132,137],[132,82],[276,164],[276,219],[228,246],[180,219],[84,164],[84,219],[132,246],[180,273],[180,110],[132,191],[228,191],[276,110]];
  const edges=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[3,6],[6,7],[7,8],[8,9],[9,3],[4,10],[10,11],[11,12],[12,9],[8,13],[1,14],[14,5],[2,16],[16,15],[15,4],[7,17],[17,1]];
  const bonds=edges.map(([a,b],i)=>`<line class="bond" x1="${points[a][0]}" y1="${points[a][1]}" x2="${points[b][0]}" y2="${points[b][1]}" style="opacity:${.58+(i%4)*.1}"/>`).join('');
  const atoms=points.map(([x,y],i)=>`<circle class="atom ${i%7===0?'o':i%5===0?'n':''}" cx="${x}" cy="${y}" r="${i%5===0?8:6}"/>`).join('');
  $('#molecule').innerHTML=`<g>${bonds}${atoms}</g>`;
  updateMoleculeTransform();
}
function updateMoleculeTransform(){ $('#molecule').style.transform=`scale(${zoom}) rotate(${rotation}deg)`; }
count.addEventListener('change',()=>{selected=Math.min(selected,Number(count.value)-1);renderCandidates();});
$('#download-report').addEventListener('click',()=>{
  const report={title:'SIRIUS interactive demo report',notice:'SIMULATED DEMO DATA ONLY. No model generation or docking was performed. Molecule diagrams are decorative and do not represent the listed SMILES.',settings:{candidate_count:Number(count.value),protein_states:Number(state.value),vina_exhaustiveness:Number(exhaustiveness.value)},candidates:examples.slice(0,Number(count.value)).map((candidate,index)=>({rank:index+1,...candidate}))};
  const blob=new Blob([JSON.stringify(report,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download='sirius-demo-report.json';link.click();URL.revokeObjectURL(url);
});
const moleculeWindow=$('#molecule-window');
moleculeWindow.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.45,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));updateMoleculeTransform();},{passive:false});
moleculeWindow.addEventListener('pointerdown',e=>{drag={x:e.clientX,last:e.clientX};moleculeWindow.setPointerCapture(e.pointerId);});
moleculeWindow.addEventListener('pointermove',e=>{if(!drag)return;rotation+=(e.clientX-drag.last)*.35;drag.last=e.clientX;updateMoleculeTransform();});
for(const event of ['pointerup','pointercancel','lostpointercapture'])moleculeWindow.addEventListener(event,()=>drag=null);

const pipeline=[
  {label:'Simulating aSAM receptor states…',step:0},
  {label:'Simulating pharmacophore generation…',step:1},
  {label:'Simulating candidate consensus…',step:2},
  {label:'Simulating CMD-Gen molecule generation…',step:3},
  {label:'Simulating Vina score ranking…',step:4}
];
$('#run-button').addEventListener('click',()=>{
  clearInterval(runTimer);
  const section=$('#run');section.hidden=false;section.scrollIntoView({behavior:'smooth'});
  $('#run-title').textContent='Simulating a run';$('#run-button').disabled=true;
  $('#run-button').innerHTML='Simulation in progress…';
  let progress=0;
  runTimer=setInterval(()=>{
    progress=Math.min(progress+Math.ceil(Math.random()*9),100);
    const step=Math.min(4,Math.floor(progress/20));
    $('#percent').textContent=`${progress}%`;
    $('#progress-fill').style.width=`${progress}%`;
    $('#run-message').textContent=progress===100?'Demo simulation complete. The examples below are preloaded sample data.':pipeline[step].label;
    $('#steps').querySelectorAll('li').forEach((li,i)=>{li.classList.toggle('done',progress===100||i<step);li.classList.toggle('active',progress<100&&i===step);});
    if(progress===100){clearInterval(runTimer);$('#run-title').textContent='Demo complete';$('#run-button').disabled=false;$('#run-button').innerHTML='Run demo again <span>→</span>';}
  },360);
});

renderCandidates();
