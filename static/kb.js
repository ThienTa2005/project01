/* Bổ sung KB / RAG / Multi-Agent. Không chèn câu hỏi người dùng qua innerHTML. */
(() => {
  'use strict';
  const el = id => document.getElementById(id);
  let EXTRA = [], KB_GRAPH = null, CHAT_HISTORY = [];
  function lineText(parent, tag, content, css='') {
    const node=document.createElement(tag);node.textContent=String(content ?? '');
    if(css)node.className=css;parent.appendChild(node);return node;
  }
  function clear(node){while(node.firstChild)node.removeChild(node.firstChild)}
  function nowLabel(){
    return new Intl.DateTimeFormat('vi-VN',{hour:'2-digit',minute:'2-digit'}).format(new Date());
  }
  function scrollChat(){
    const box=el('kb-chat-messages');if(box)requestAnimationFrame(()=>{box.scrollTop=box.scrollHeight});
  }
  function chatBubble(role, content, note=''){
    const box=el('kb-chat-messages');if(!box)return null;
    const row=document.createElement('div');
    row.className='message-row '+(role==='user'?'user-message':'assistant-message');
    if(role==='error')row.className='message-row assistant-message error-message';
    if(role!=='user'){
      const avatar=document.createElement('div');avatar.className='message-avatar';
      avatar.innerHTML='<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M8 11h16a4 4 0 0 1 4 4v8a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4v-8a4 4 0 0 1 4-4Z"/><path d="M16 5v6M10 19h.01M16 19h.01M22 19h.01"/></svg>';
      row.appendChild(avatar);
    }
    const stack=document.createElement('div');stack.className='message-stack';
    const meta=document.createElement('div');meta.className='message-meta';
    lineText(meta,'b',role==='user'?'Bạn':'DiaBot');lineText(meta,'span',nowLabel());
    const bubble=document.createElement('div');bubble.className='message-bubble';
    lineText(bubble,'p',content);
    if(note)lineText(bubble,'small',note,'message-note');
    stack.append(meta,bubble);row.appendChild(stack);box.appendChild(row);scrollChat();return row;
  }
  function showTyping(){
    const box=el('kb-chat-messages');if(!box)return null;
    const row=document.createElement('div');row.className='message-row assistant-message';row.id='kb-typing';
    const avatar=document.createElement('div');avatar.className='message-avatar';
    avatar.innerHTML='<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M8 11h16a4 4 0 0 1 4 4v8a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4v-8a4 4 0 0 1 4-4Z"/><path d="M16 5v6M10 19h.01M16 19h.01M22 19h.01"/></svg>';
    const stack=document.createElement('div');stack.className='message-stack';
    const meta=document.createElement('div');meta.className='message-meta';lineText(meta,'b','DiaBot');lineText(meta,'span','đang truy xuất tri thức…');
    const bubble=document.createElement('div');bubble.className='message-bubble typing-bubble';
    for(let i=0;i<3;i++)bubble.appendChild(document.createElement('i'));
    stack.append(meta,bubble);row.append(avatar,stack);box.appendChild(row);scrollChat();return row;
  }
  async function api(url, options={}){
    const r=await fetch(url,options); const d=await r.json();
    if(!r.ok)throw new Error(d.error || `Lỗi HTTP ${r.status}`);return d;
  }
  function aLink(parent, url, label){
    const link=document.createElement('a');link.textContent=label;
    if(url && /^https:\/\//i.test(url)){link.href=url;link.target='_blank';link.rel='noopener noreferrer'}
    parent.appendChild(link);return link;
  }
  function graphView(box, edges, focus='', max=40){
    clear(box);
    if(!edges || edges.length===0){lineText(box,'p','Chưa có cạnh đồ thị trong kết quả.','small');return}
    const chosen=focus?edges.filter(e=>e.source===focus||e.target===focus):edges.slice(0,max);
    const displayed=(chosen.length?chosen:edges).slice(0,max);
    const names=[...new Set(displayed.flatMap(e=>[e.source,e.target]))];
    if(names.length<1)return;
    const svgNS='http://www.w3.org/2000/svg';
    const node=(tag, attrs, parent)=>{const x=document.createElementNS(svgNS,tag);for(const k in attrs)x.setAttribute(k,attrs[k]);(parent||svg).appendChild(x);return x};
    const svg=document.createElementNS(svgNS,'svg');svg.setAttribute('viewBox','0 0 960 570');svg.setAttribute('role','img');svg.setAttribute('aria-label','Đồ thị thực thể và quan hệ');
    const center=names.includes(focus)?focus:names[0];
    const pos={};pos[center]=[480,280];
    const rest=names.filter(x=>x!==center);
    rest.forEach((n,i)=>{const theta=2*Math.PI*i/Math.max(rest.length,1)-Math.PI/2;
      const rad=210+(i%3)*10;pos[n]=[480+Math.cos(theta)*rad*1.65,285+Math.sin(theta)*rad]});
    displayed.forEach(e=>{
      const a=pos[e.source], b=pos[e.target];if(!a||!b)return;
      const ln=node('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'#7295b8','stroke-width':'1.8'},svg);
      const title=node('title',{},ln);title.textContent=e.relation+(e.source_id?' · '+e.source_id:'');
    });
    names.forEach(name=>{
      const [x,y]=pos[name];const g=node('g',{style:'cursor:pointer'},svg);
      node('circle',{cx:x,cy:y,r:name===center?25:17,fill:name===center?'#1d6fd6':'#e0eefb',stroke:'#2a6ead','stroke-width':'1.5'},g);
      const t=node('text',{x,y:y+(name===center?43:33),'text-anchor':'middle',fill:'currentColor','font-size':name.length>25?'10':'12'},g);
      t.textContent=name.length>40?name.slice(0,38)+'…':name;
      const title=node('title',{},g);title.textContent=name;
      g.addEventListener('click',()=>{const sel=el('kb-focus');if(sel && [...sel.options].some(o=>o.value===name)){sel.value=name;renderFullGraph()}});
    });
    box.appendChild(svg);
  }
  async function kbInfo(){
    const d=await api('/api/kb');
    const info=el('kb-stats');clear(info);
    [['Thực thể',d.stats.entities],['Quan hệ',d.stats.relations],['Nguồn',d.stats.sources]].forEach(([n,v])=>{
      const card=document.createElement('div');card.className='metric-stat';
      lineText(card,'strong',v);lineText(card,'span',n);info.appendChild(card);
    });
    const sources=el('kb-reference-list');clear(sources);
    d.sources.forEach(s=>{
      const div=document.createElement('div');div.className='well';div.style.marginBottom='8px';
      lineText(div,'strong',`${s.id} · ${s.title}`);lineText(div,'p',s.summary,'small');aLink(div,s.url,s.url);sources.appendChild(div);
    });
    const status=await api('/api/kb/neo4j');
    el('neo4j-status').textContent=status.connected
      ?`Neo4j đã kết nối: ${status.entity_nodes} thực thể / ${status.relationships} quan hệ (database).`
      :`Neo4j chưa hoạt động trên server; dùng NetworkX. ${status.note||''}`;
    el('neo4j-status').classList.toggle('success',!!status.connected);
    el('neo4j-status').classList.toggle('neutral',!status.connected);
  }
  async function loadKbGraph(){
    if(!KB_GRAPH){KB_GRAPH=await api('/api/kb/graph');
      const sel=el('kb-focus');const entities=KB_GRAPH.nodes.filter(n=>n.kind==='Entity');
      const already=new Set([...sel.options].map(o=>o.value));
      entities.forEach(n=>{if(!already.has(n.id))sel.add(new Option(n.id,n.id))});
      await kbInfo();}
    renderFullGraph();
  }
  window.loadKbGraph=()=>loadKbGraph().catch(err=>el('neo4j-status').textContent=err.message);
  function renderFullGraph(){
    const focus=el('kb-focus').value;
    const matches=KB_GRAPH.edges.filter(e=>e.source===focus||e.target===focus);
    el('kb-graph-count').textContent=`Tổng ${KB_GRAPH.edges.length} quan hệ, đang xem ${matches.length} quan hệ quanh “${focus}”. Chọn thực thể khác để khám phá.`;
    graphView(el('kb-fullgraph'),KB_GRAPH.edges,focus,38);
  }
  el('kb-focus').addEventListener('change',()=>{if(KB_GRAPH)renderFullGraph()});
  el('kb-path-go').onclick=async()=>{
    const d=await api('/api/kb/path?'+new URLSearchParams({start:el('kb-start').value,end:el('kb-end').value,hops:'5'}));
    el('kb-path-result').textContent=d.found?`${d.hops} chặng: ${d.path.join(' → ')}`:d.reason;
    if(d.found)graphView(el('kb-fullgraph'),d.edges,'',10);
  };
  function setQuestion(question){
    const input=el('kb-q');input.value=question;input.style.height='auto';input.style.height=Math.min(input.scrollHeight,120)+'px';input.focus();
  }
  document.querySelectorAll('[data-chat-example]').forEach(button=>{
    button.addEventListener('click',()=>setQuestion(button.dataset.chatExample));
  });
  el('kb-example').onclick=()=>setQuestion('Vì sao đái tháo đường gây biến chứng thận, cần kiểm tra gì?');
  el('kb-q').addEventListener('input',event=>{
    event.target.style.height='auto';event.target.style.height=Math.min(event.target.scrollHeight,120)+'px';
  });
  el('kb-q').addEventListener('keydown',event=>{
    if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();el('kb-go').click()}
  });
  el('kb-go').onclick=async()=>{
    el('kb-error').textContent='';const btn=el('kb-go');
    const question=el('kb-q').value.trim();
    if(!question){el('kb-error').textContent='Hãy nhập câu hỏi trước khi gửi.';el('kb-q').focus();return}
    btn.disabled=true;chatBubble('user',question);el('kb-q').value='';el('kb-q').style.height='auto';showTyping();
    try{
      const include=el('kb-include-patient').checked;
      const body={question,method:el('kb-method').value,top_k:Number(el('kb-topk').value),
          hops:Number(el('kb-hops').value),alpha:Number(el('kb-alpha').value),
          extra_documents:EXTRA,include_patient:include,patient:include?collect():null,
          history:CHAT_HISTORY.slice(-6)};
      const d=await api('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      CHAT_HISTORY.push({role:'user',content:body.question},{role:'assistant',content:d.answer});
      CHAT_HISTORY=CHAT_HISTORY.slice(-12);
      const context=d.context_used?' · đã hiểu theo câu hỏi trước':'';
      el('kb-typing')?.remove();
      const fallbackLabels={
        disabled:'Gemini chưa bật',missing_api_key:'thiếu API key',sdk_not_installed:'thiếu Google SDK',
        no_retrieved_evidence:'không có bằng chứng',missing_citation:'Gemini thiếu trích dẫn',
        unknown_citation:'Gemini tạo nguồn không hợp lệ'
      };
      const fallback=d.llm?.fallback_reason||'';
      const fallbackText=fallback.startsWith('provider_error:')
        ?'lỗi kết nối/quota Gemini'
        :(fallbackLabels[fallback]||'RAG offline');
      const generator=d.llm?.used?`Gemini · ${d.llm.model}`:`Offline · ${fallbackText}`;
      chatBubble('assistant',d.answer,`${d.sources.length} nguồn · ${d.graph_backend||'NetworkX'} · ${generator}`);
      el('kb-answer').textContent=d.answer+'\n\n(Ý định: '+(d.intent||'general')+context+' · KG backend: '+(d.graph_backend||'NetworkX')+')';
      const sn=el('kb-snippets');clear(sn);
      d.snippets.forEach(s=>{const w=document.createElement('div');w.className='well';w.style.margin='6px 0';
        lineText(w,'div',`${s.source_id} · ${s.kind==='fact'?'Cạnh tri thức':'Tài liệu'} · Điểm ${s.score}`,'small');lineText(w,'p',s.text);sn.appendChild(w)});
      const source=el('kb-sources');clear(source);
      d.sources.forEach(s=>{const li=document.createElement('li');lineText(li,'strong',`${s.id} · ${s.title} — `);
        if(s.url)aLink(li,s.url,'Mở nguồn gốc');else lineText(li,'em','Nguồn file người dùng (không có URL)');source.appendChild(li)});
      graphView(el('kb-subgraph'),d.subgraph.edges,'',22);
      const trace=el('kb-trace');clear(trace);
      d.trace.forEach(x=>{const w=document.createElement('div');w.className='well';
        lineText(w,'strong',x.agent);lineText(w,'p',x.action);lineText(w,'span',x.detail||'','small');trace.appendChild(w)});
      const mas=el('kb-mas');clear(mas);
      if(d.decision_support){
        mas.classList.remove('hide');const ds=d.decision_support, ar=ds.agent_results||{};
        lineText(mas,'strong',`MAS runtime · ${ds.request_id} · ${ds.status}`);
        const dl=ar.DLPredictionAgent;
        if(dl && dl.status==='ok')lineText(mas,'p',`Deep Learning: điểm ${(dl.payload.score*100).toFixed(1)}% · ngưỡng ${(dl.payload.threshold*100).toFixed(1)}% · ${dl.payload.high_risk_flag?'cao hơn ngưỡng':'thấp hơn ngưỡng'}`);
        else if(dl)lineText(mas,'p',`Deep Learning: ${dl.error?.message||dl.status}`);
        const rules=ar.RuleBasedAgent?.payload?.flags||[];
        lineText(mas,'p',`Rule-Based Agent: ${rules.length} cờ · ${ds.report.human_review_required?'cần người có chuyên môn xem xét':'không có cờ bắt buộc xem xét'}`);
        rules.forEach(x=>lineText(mas,'div',`${x.severity.toUpperCase()} · ${x.message}`,'small'));
        lineText(mas,'p','Audit chỉ lưu trạng thái, phiên bản và thời gian agent; không lưu đặc trưng bệnh nhân.','small');
      }else mas.classList.add('hide');
      const ml=el('kb-ml');clear(ml);
      if(d.ml_prediction){ml.classList.remove('hide');lineText(ml,'strong','Clinical ML Predictor: kết quả hội đồng');
        lineText(ml,'p',`${d.ml_prediction.votes_high}/${d.ml_prediction.votes_total} agent cảnh báo nguy cơ cao. Đây là đánh giá thuật toán, không chẩn đoán.`);
        d.ml_prediction.results.forEach(r=>lineText(ml,'div',`${r.label}: ${r.status==='ok'?(r.probability*100).toFixed(1)+'%':r.reason||r.status}`,'small'));
      }else ml.classList.add('hide');
      el('kb-empty-insight')?.classList.add('hide');
      el('kb-result').classList.remove('hide');
    }catch(err){
      el('kb-typing')?.remove();el('kb-error').textContent=err.message;chatBubble('error','Xin lỗi, tôi chưa thể xử lý câu hỏi này. '+err.message);
      setQuestion(question);
    }finally{el('kb-typing')?.remove();btn.disabled=false}
  };
  function showSession(){
    const box=el('kb-session-sources');clear(box);
    if(!EXTRA.length){lineText(box,'span','Chưa có tài liệu tạm.','small');return}
    EXTRA.forEach((doc,i)=>{lineText(box,'p',`${i+1}. ${doc.title} — ${doc.mentions.join(', ')||'Chưa nhận diện được thực thể'}`,'small')});
    lineText(box,'p','Tài liệu này sẽ được gửi kèm câu hỏi trong tab KB + RAG; không ghi vào nguồn KB gốc.','small');
  }
  el('kb-clear-docs').onclick=()=>{EXTRA=[];showSession()};
  el('kb-ingest').onclick=async()=>{
    el('kb-ingest-error').textContent='';const btn=el('kb-ingest');btn.disabled=true;
    try{
      if(EXTRA.length>=3)throw new Error('Tối đa 3 tài liệu tạm. Xóa bớt trước khi tiếp tục.');
      const file=el('kb-new-pdf').files[0];let data;
      if(file){const fd=new FormData();fd.append('file',file);data=await api('/api/kb/ingest',{method:'POST',body:fd});}
      else if(el('kb-new-url').value.trim())data=await api('/api/kb/ingest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:el('kb-new-url').value.trim()})});
      else throw new Error('Hãy nhập URL nguồn hoặc chọn PDF.');
      EXTRA.push(data);showSession();
    }catch(err){el('kb-ingest-error').textContent=err.message}finally{btn.disabled=false}
  };
  api('/api/llm/gemini').then(status=>{
    const badge=el('gemini-status');if(!badge)return;
    badge.textContent=status.available?'Gemini sẵn sàng':'Gemini offline';
    badge.classList.toggle('gemini-on',!!status.available);
    badge.title=status.available
      ?`Explanation Agent: ${status.model}`
      :status.configured?'Đã có API key nhưng GEMINI_ENABLED chưa bật hoặc SDK chưa sẵn sàng':'Đang dùng câu trả lời RAG offline';
  }).catch(()=>{});
})();
