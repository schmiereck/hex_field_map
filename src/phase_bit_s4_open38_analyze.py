#!/usr/bin/env python3
"""Bit19 reproducible secondary diagnostics and compact SVG figure.

No new dynamics: uses the JSON produced by phase_bit_s4_open38_phased_packet.
Early motion fits are qualitative and intentionally NOT treated as certified
asymptotic velocities (two seeds, periodic finite box).
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
import numpy as np


def summarize(raw):
    rows=[]
    for mode in ('0','2','-2'):
        collision=raw['measurements'][mode]
        free=raw['free_streaming_control'][mode]
        ch={item['channel']:item for item in collision['channels']}
        ctrl={item['channel']:item for item in free['channels']}
        q=ch['hidden_fourier_q']
        x=ch['visible_w0']
        avg=ch['unweighted_w_mean']
        t=np.arange(2,9)
        rows.append(dict(mode=int(mode),initial_visible_w0_amplitude=x['initial_amplitude'],
          initial_q_amplitude=q['initial_amplitude'],
          initial_w_average_amplitude=avg['initial_amplitude'],
          initial_w_mean_to_q_amplitude_ratio=avg['initial_amplitude']/q['initial_amplitude'],
          q_amplitude_ratio_at_tick_8=q['demod_amplitude_ratio'][8],
          q_amplitude_ratio_at_tick_24=q['demod_amplitude_ratio'][24],
          q_amplitude_ratio_free_at_tick_24=ctrl['hidden_fourier_q']['demod_amplitude_ratio'][24],
          visible_w0_amplitude_ratio_tick24=x['demod_amplitude_ratio'][24],
          q_width_initial=q['width'][0], q_width_tick8=q['width'][8], q_width_tick24=q['width'][24],
          q_center_early_slope_tick2_to8=float(np.polyfit(t,np.array(q['center'])[t],1)[0]),
          visible_center_early_slope_tick2_to8=float(np.polyfit(t,np.array(x['center'])[t],1)[0]),
          q_phase_early_slope_tick2_to8=float(np.polyfit(t,np.array(q['demod_phase'])[t],1)[0]),
          q_phase_early_slope_free_tick2_to8=float(np.polyfit(t,np.array(ctrl['hidden_fourier_q']['demod_phase'])[t],1)[0]),
          q_signal_variance_ratio_tick24=collision['hidden_q_coherent_power_over_seed_variance_by_tick'][24]))
    return {'source':'data/bit19_phasecoded_main.json','fit_window':'ticks 2..8 inclusive; descriptive, NOT robust inferred group velocities',
        'comparison':'same Bit8 collision vs pure permutation stream for every real trajectory',
        'symmetry':'independent x-mirror microstate control exact on five forward ticks and reverse',
        'no_particle_claim':True,'no_eigenmode_claim':True,'only_two_independent_seeds':True,
        'no_evidence_of_exact_shell_closure':True,'rows':rows}


def svg_chart(raw, svg_path):
    W,H=760,430
    margin=(85,35,54,62)
    x0,y0=margin[0],margin[1]
    pw,ph=W-margin[0]-margin[2],H-margin[1]-margin[3]
    def cx(t):return x0+pw*t/24
    def cy(v):return y0+ph*(1-max(0.,min(1.,v/1.05)))
    import html
    colors={'0':'#7b58be','2':'#198b91','-2':'#d4882f'}
    legend={'0':'k=0','2':'k=+π/8','-2':'k=−π/8'}
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
           '<title id="title">Bit19: Kohärente Mode-Amplituden im verborgenen Fourierkanal</title>',
           '<desc id="desc">Normierte Amplituden für Ruhe-, positiven und negativen Träger, exakt symmetrische Kollision durchgezogen und freier Transport gestrichelt.</desc>',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="75" y="20" font-size="16" font-family="sans-serif" fill="#222">4D-Bitfeld: Kohärente q-Trägeramplitude</text>']
    for val in [0,.25,.5,.75,1.]:
        y=cy(val)
        lines.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" stroke="#e6e9ee"/>')
        lines.append(f'<text x="{x0-12}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#333">{val:.2f}</text>')
    for t in [0,4,8,12,16,20,24]:
        xpos=cx(t)
        lines.append(f'<text x="{xpos:.1f}" y="{y0+ph+22}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#333">{t}</text>')
    for mode in ('0','2','-2'):
        for group, dash in [('free_streaming_control',' stroke-dasharray="7 4"'),('measurements','')]:
            ch=next(c for c in raw[group][mode]['channels'] if c['channel']=='hidden_fourier_q')
            vals=ch['demod_amplitude_ratio']
            pts=' '.join(f'{cx(t):.2f},{cy(v):.2f}' for t,v in enumerate(vals))
            lines.append(f'<polyline points="{pts}" fill="none" stroke="{colors[mode]}" stroke-width="2.5" opacity="0.85"{dash}/>')
    lines.append(f'<text x="{x0+pw/2}" y="{H-14}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#333">Tick</text>')
    lines.append(f'<text x="20" y="{y0+ph/2}" font-size="12" font-family="sans-serif" fill="#333" transform="rotate(-90 20 {y0+ph/2})" text-anchor="middle">Trägeramplitude / Anfang</text>')
    for j,mode in enumerate(('0','2','-2')):
        xx=170+j*175; yy=H-35
        lines.extend((f'<line x1="{xx}" y1="{yy}" x2="{xx+25}" y2="{yy}" stroke="{colors[mode]}" stroke-width="3"/>',
                      f'<text x="{xx+32}" y="{yy+4}" font-family="sans-serif" font-size="12" fill="#333">{html.escape(legend[mode])}</text>'))
    lines.append('<text x="505" y="49" font-size="11" fill="#333" font-family="sans-serif">durchgezogen: Bit8 · gestrichelt: frei</text>')
    lines.append('</svg>')
    svg_path.parent.mkdir(parents=True,exist_ok=True)
    svg_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--figure',type=Path,required=True)
    a=ap.parse_args();raw=json.loads(a.input.read_text());summ=summarize(raw)
    a.summary.parent.mkdir(parents=True,exist_ok=True)
    a.summary.write_text(json.dumps(summ,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    svg_chart(raw,a.figure)
    for r in summ['rows']:
        print(r['mode'],'early vq',round(r['q_center_early_slope_tick2_to8'],3),
              'late q amp',round(r['q_amplitude_ratio_at_tick_24'],3),
              'width',round(r['q_width_initial'],2),'->',round(r['q_width_tick24'],2))