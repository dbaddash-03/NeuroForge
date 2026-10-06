const views = ["overview","pipeline","playground","architecture","training","experiments"];
const titles = {
  overview:"LLM From Scratch Studio", pipeline:"The Learning Pipeline", playground:"Model Playground",
  architecture:"Transformer Anatomy", training:"Training Laboratory", experiments:"Experiment Board"
};

function setView(view){
  document.querySelectorAll(".view").forEach(v=>v.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(b=>b.classList.remove("active"));
  document.getElementById(view).classList.add("active");
  document.querySelector(`[data-view="${view}"]`).classList.add("active");
  document.getElementById("page-title").textContent=titles[view];
  window.scrollTo({top:0,behavior:"smooth"});
}
window.setView=setView;

document.querySelectorAll(".nav-item").forEach(btn=>{
  btn.addEventListener("click",()=>setView(btn.dataset.view));
});

document.getElementById("themeBtn").addEventListener("click",()=>{
  document.body.classList.toggle("dark-mode");
});

const temp = document.getElementById("temperature");
const tokens = document.getElementById("tokens");
temp.addEventListener("input",()=>document.getElementById("tempValue").textContent=temp.value);
tokens.addEventListener("input",()=>document.getElementById("tokenValue").textContent=tokens.value);

async function generate(){
  const output = document.getElementById("output");
  const tag = document.getElementById("modeTag");
  const btn = document.getElementById("generateBtn");
  btn.disabled=true; btn.innerHTML="GENERATING <span>…</span>";
  tag.textContent="RUNNING";
  output.textContent="Sampling next-token probabilities…";
  try{
    const res=await fetch("/api/generate",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        prompt:document.getElementById("prompt").value,
        max_new_tokens:Number(tokens.value),
        temperature:Number(temp.value),
        top_k:40
      })
    });
    const data=await res.json();
    output.textContent=data.text;
    tag.textContent=(data.mode||"LOCAL").toUpperCase();
  }catch(e){
    output.textContent="Backend unavailable. Start uvicorn and reload the studio.";
    tag.textContent="OFFLINE";
  }finally{
    btn.disabled=false; btn.innerHTML="GENERATE <span>↗</span>";
  }
}
document.getElementById("generateBtn").addEventListener("click",generate);
function scrollToPlayground(){setView("playground")}
window.scrollToPlayground=scrollToPlayground;
