def run_pretrained():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = os.environ.get("ERIS_PT", "facebook/bart-base")
    tkz = AutoTokenizer.from_pretrained(name)
    md = AutoModelForSeq2SeqLM.from_pretrained(name)
    md.to(DEVICE)
    dm = md.config.d_model
    thead = nn.Linear(dm, NTERM).to(DEVICE)
    with torch.no_grad():
        pri = np.clip(term_prior, 1e-4, 1 - 1e-4)
        thead.bias.copy_(torch.tensor(np.log(pri / (1 - pri)), dtype=torch.float32))
        thead.weight.mul_(0.01)
    print(f"[pt] {name} {sum(p.numel() for p in md.parameters())/1e6:.0f}M t={time.time()-T_START:.0f}s", flush=True)

    def enc_all(texts, maxlen):
        return tkz(list(texts), truncation=True, max_length=maxlen)["input_ids"]
    SX = enc_all(train["reasoning_trace"], PT_SRC)
    SY = enc_all(train["canonical_restatement"], PT_TGT)
    TX = enc_all(test["reasoning_trace"], PT_SRC)
    pad = tkz.pad_token_id

    def pad_batch(seqs):
        L = max(len(s) for s in seqs)
        a = np.full((len(seqs), L), pad, dtype=np.int64)
        for i, s in enumerate(seqs):
            a[i, : len(s)] = s
        t = torch.from_numpy(a)
        return t, (t != pad).long()

    def make_bl(ids, bs):
        ids = np.array(ids)
        o = ids[np.argsort([len(SX[i]) for i in ids], kind="stable")]
        return [o[i:i + bs] for i in range(0, len(o), bs)]

    params = list(md.parameters()) + list(thead.parameters())
    opt = torch.optim.AdamW(params, lr=PT_LR, weight_decay=0.01)
    t_end = T_START + BUDGET * PT_FRAC
    t0 = time.time(); span = max(1.0, t_end - t0); step = 0; ep = 0
    warm = 40
    while time.time() < t_end:
        md.train(); bl = make_bl(trn_idx, PT_BS); np.random.shuffle(bl)
        run = n = 0.
        for bidx in bl:
            frac = min(1.0, (time.time() - t0) / span)
            lr = PT_LR * (step + 1) / warm if step < warm else 1e-6 + 0.5 * (PT_LR - 1e-6) * (1 + math.cos(math.pi * frac))
            for gg in opt.param_groups:
                gg["lr"] = lr
            xi, xm = pad_batch([SX[i] for i in bidx])
            yi, _ = pad_batch([SY[i] for i in bidx])
            lab = yi.clone(); lab[lab == pad] = -100
            eo = md.model.encoder(input_ids=xi, attention_mask=xm)
            out = md(attention_mask=xm, encoder_outputs=eo, labels=lab)
            h = eo.last_hidden_state
            mm = xm.unsqueeze(-1).float()
            pooled = (h * mm).sum(1) / mm.sum(1).clamp(min=1)
            ty = np.zeros((len(bidx), NTERM), dtype=np.float32)
            for b, i in enumerate(bidx):
                for w in INTRO[i]:
                    if w in tix:
                        ty[b, tix[w]] = 1.
            tl = F.binary_cross_entropy_with_logits(thead(pooled), torch.from_numpy(ty)) * NTERM / 20.
            loss = out.loss + 0.3 * tl
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); step += 1
            run += float(out.loss); n += 1
            if step == 1:
                print(f"[pt] first-step ce {float(out.loss):.3f}", flush=True)
            if time.time() >= t_end:
                break
        ep += 1
        print(f"[pt] epoch {ep} ce {run/max(1,n):.3f} lr {lr:.2e} t={time.time()-T_START:.0f}s", flush=True)
    if ep == 0:
        raise RuntimeError("no pretrained epoch completed")

    md.eval(); thead.eval()

    @torch.no_grad()
    def pt_probs(seqs):
        out = np.zeros((len(seqs), NTERM), dtype=np.float32)
        for s0 in range(0, len(seqs), 16):
            xi, xm = pad_batch(seqs[s0:s0 + 16])
            h = md.model.encoder(input_ids=xi, attention_mask=xm).last_hidden_state
            mm = xm.unsqueeze(-1).float()
            out[s0:s0 + 16] = torch.sigmoid(thead((h * mm).sum(1) / mm.sum(1).clamp(min=1))).numpy()
        return out

    @torch.no_grad()
    def pt_gen(seqs, nb=4, minl=30, maxl=80, lp=1.0, bs=16):
        res = []
        for s0 in range(0, len(seqs), bs):
            xi, xm = pad_batch(seqs[s0:s0 + bs])
            o = md.generate(input_ids=xi, attention_mask=xm, num_beams=nb, min_length=minl,
                            max_length=maxl, length_penalty=lp, early_stopping=True,
                            no_repeat_ngram_size=0)
            res += tkz.batch_decode(o, skip_special_tokens=True)
        return res

    te_p = pt_probs(TX)
    te_mask_p = np.ones_like(te_p)
    for j in range(len(TX)):
        for w in te_sc[j]:
            if w in tix:
                te_mask_p[j, tix[w]] = 0.
    te_p *= te_mask_p
    te_txt = pt_gen(TX)
    print(f"[pt] test decoded t={time.time()-T_START:.0f}s", flush=True)

    def emit(kk, bb, lpen_txt):
        rows = []
        for j in range(len(TX)):
            ws = tok(lpen_txt[j])
            txt = " ".join(ws).strip() or FALLBACK_TXT
            p = (1 - bb) * te_p[j] + bb * term_prior * te_mask_p[j]
            P = sorted(terms[x] for x in pick_terms(p, kk))
            rows.append({"id": test["id"].iloc[j], "predicted_restatement": txt,
                         "introduced_terms": ";".join(P) if P else FALLBACK_TERMS})
        s = sample[["id"]].merge(pd.DataFrame(rows), on="id", how="left")
        s["predicted_restatement"] = s["predicted_restatement"].fillna(FALLBACK_TXT)
        s["introduced_terms"] = s["introduced_terms"].fillna(FALLBACK_TERMS)
        s.to_csv(submission_out, index=False)
        return s

    s = emit(K_DEFAULT, 0.0, te_txt)
    print(f"[pt] baseline secured {s.shape} t={time.time()-T_START:.0f}s", flush=True)

    try:
        if time.time() - T_START < BUDGET * 0.93 and len(val_idx):
            vl = list(val_idx)
            vp2 = pt_probs([SX[i] for i in vl])
            vmask = np.ones_like(vp2)
            for j, i in enumerate(vl):
                for w in tr_sc[i]:
                    if w in tix:
                        vmask[j, tix[w]] = 0.
            vp2 *= vmask
            vtxt = pt_gen([SX[i] for i in vl])
            G = float(np.mean([rl(tok(vtxt[j]), tr_tgt[i]) for j, i in enumerate(vl)]))
            print(f"[pt] holdout G={G:.4f} len={np.mean([len(tok(t)) for t in vtxt]):.1f} "
                  f"t={time.time()-T_START:.0f}s", flush=True)
            bk, bb, bv = K_DEFAULT, 0.0, -1.
            for kk in [0, 10, 12, 13, 14, 15, 16, 18]:
                for bb_ in [0.0, 0.15, 0.3, 0.45]:
                    sc = []
                    for j, i in enumerate(vl):
                        p = (1 - bb_) * vp2[j] + bb_ * term_prior * vmask[j]
                        P = set(terms[x] for x in pick_terms(p, kk))
                        Tt = INTRO[i]
                        sc.append(0. if (not P or not Tt) else 2 * len(P & Tt) / (len(P) + len(Tt)))
                    v = float(np.mean(sc))
                    if v > bv:
                        bv, bk, bb = v, kk, bb_
            print(f"[pt] holdout V={bv:.4f} k={bk} blend={bb}", flush=True)
            print(f"[pt] holdout TSS={0.6*G + 0.4*bv:.4f}", flush=True)
            s = emit(bk, bb, te_txt)
    except Exception as _e:
        print("[pt] tuning skipped:", repr(_e), flush=True)
    print("wrote", submission_out, s.shape, f"total {time.time()-T_START:.0f}s", flush=True)
    return True
