import { useEffect, useMemo, useState } from 'react'

const defaultTeams = ['Chennai Super Kings', 'Mumbai Indians']

function TeamSelect({ label, value, teams, onChange, excluded }) {
  return <label className="team-select">
    <span>{label}</span>
    <select value={value} onChange={(event) => onChange(event.target.value)}>
      {teams.filter((team) => team !== excluded).map((team) => <option key={team}>{team}</option>)}
    </select>
  </label>
}

function Probability({ team, value, accent }) {
  return <section className="probability-card" style={{ '--accent': accent }}>
    <p>{team}</p>
    <strong>{(value * 100).toFixed(1)}<small>%</small></strong>
    <div className="meter"><i style={{ width: `${value * 100}%` }} /></div>
  </section>
}

function FormGuide({ form }) {
  return <div className="form-guide">
    {form.map((result, i) => <span key={i} className={`form-badge ${result}`}>{result}</span>)}
  </div>
}

function H2HSection({ data }) {
  if (!data) return null;
  const aWinPct = data.h2h.matches > 0 ? (data.h2h.team_a_wins / data.h2h.matches * 100) : 50;
  
  return <section className="panel h2h-panel">
    <div className="panel-heading"><span>02</span><div><p>HISTORICAL CONTEXT</p><h2>Head-to-Head & Form Guide</h2></div></div>
    
    <div className="h2h-content">
      <div className="h2h-team">
        <p>{data.team_a}</p>
        <FormGuide form={data.form_a} />
      </div>
      
      <div className="h2h-center">
        <div className="h2h-stats">
          <b>{data.h2h.team_a_wins}</b>
          <small>{data.h2h.matches} MATCHES</small>
          <b>{data.h2h.team_b_wins}</b>
        </div>
        <div className="h2h-bar">
          <i style={{ width: `${aWinPct}%` }} />
        </div>
      </div>
      
      <div className="h2h-team right">
        <p>{data.team_b}</p>
        <FormGuide form={data.form_b} />
      </div>
    </div>
  </section>
}


function App() {
  const [teams, setTeams] = useState(defaultTeams)
  const [teamA, setTeamA] = useState(defaultTeams[0])
  const [teamB, setTeamB] = useState(defaultTeams[1])
  const [report, setReport] = useState(null)
  const [h2hData, setH2hData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [seasons, setSeasons] = useState([])
  const [season, setSeason] = useState('')
  const [fixtures, setFixtures] = useState([])
  const [fixtureFilter, setFixtureFilter] = useState('all')
  const [selectedFixture, setSelectedFixture] = useState(null)
  const [theme, setTheme] = useState('aurora')

  useEffect(() => {
    fetch('/api/teams').then((response) => response.json()).then((data) => {
      if (data.teams?.length) setTeams(data.teams)
    }).catch(() => setError('Start the Python API to load the current IPL data.'))
  }, [])

  useEffect(() => {
    fetch('/api/seasons').then((response) => response.json()).then((data) => {
      setSeasons(data.seasons || [])
      if (data.seasons?.length) setSeason(data.seasons[0])
    }).catch(() => setError('Start the Python API to load season fixtures.'))
  }, [])

  useEffect(() => {
    if (!season) return
    fetch(`/api/fixtures?season=${encodeURIComponent(season)}`).then((response) => response.json()).then((data) => {
      if (!data.fixtures) throw new Error(data.detail || 'Unable to load fixtures.')
      setFixtures(data.fixtures); setSelectedFixture(null)
    }).catch((requestError) => setError(requestError.message))
  }, [season])

  const prediction = report?.prediction
  const playerTeams = useMemo(() => Object.entries(report?.inferred_player_impacts || {}), [report])
  const visibleFixtures = useMemo(() => fixtures.filter((fixture) => fixtureFilter === 'all' ||
    (fixtureFilter === 'playoffs' ? fixture.stage !== 'League stage' : fixture.stage === 'League stage')), [fixtures, fixtureFilter])

  async function predict() {
    setLoading(true); setError(''); setH2hData(null);
    try {
      const response = await fetch(`/api/insights?team_a=${encodeURIComponent(teamA)}&team_b=${encodeURIComponent(teamB)}`)
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Unable to build prediction.')
      setReport(data)

      try {
        const h2hResponse = await fetch(`/api/h2h?team_a=${encodeURIComponent(teamA)}&team_b=${encodeURIComponent(teamB)}`)
        if (h2hResponse.ok) {
          const h2hJson = await h2hResponse.json()
          setH2hData(h2hJson)
        }
      } catch (e) {
        console.error("Failed to load H2H", e)
      }
    } catch (requestError) {
      setError(requestError.message)
    } finally { setLoading(false) }
  }

  return <main className={`app ${theme}`}>
    <div className="bg-effects" aria-hidden="true">
      <div className="orb orb-1" />
      <div className="orb orb-2" />
      <div className="orb orb-3" />
      <div className="orb orb-4" />
      <div className="grid-overlay" />
      <div className="noise-overlay" />
    </div>
    <header className="hero">
      <div className="topbar"><div className="brand"><span className="ball">●</span> <em>CRICKONOMICS</em></div><div className="theme-picker" aria-label="Colour theme">{[['aurora', 'Aurora'], ['violet', 'Violet'], ['ember', 'Ember']].map(([value, label]) => <button key={value} className={theme === value ? 'active' : ''} onClick={() => setTheme(value)}>{label}</button>)}</div><span className="status"><i /> AI MODEL ONLINE</span></div>
      <div className="hero-copy"><p className="eyebrow">PRE MATCH INTELLIGENCE</p><h1>Know the game<b>before the first ball.</b></h1><p>Team form, player impact, and transparent win probabilities — in one clear match room.</p></div>
      <div className="hero-orbit"><span>DATA</span><i /><span>FORM</span><i /><span>FORESIGHT</span></div>
    </header>

    <section className="selector panel">
      <div className="panel-heading"><span>01</span><div><p>CREATE A MATCHUP</p><h2>Select two IPL teams</h2></div></div>
      <div className="selection-row">
        <TeamSelect label="TEAM A" value={teamA} teams={teams} onChange={setTeamA} excluded={teamB} />
        <div className="versus">VS</div>
        <TeamSelect label="TEAM B" value={teamB} teams={teams} onChange={setTeamB} excluded={teamA} />
        <button onClick={predict} disabled={loading}>{loading ? 'ANALYSING…' : 'ANALYSE MATCH →'}</button>
      </div>
    </section>

    {error && <div className="error">{error}</div>}
    {!report && !error && <section className="empty"><span>🏏</span><h2>Your match room is ready</h2><p>Select two teams and generate a fully explained prediction.</p></section>}

    {report && <>
      <section className="result-head">
        <div><p className="eyebrow">MODEL OUTPUT · {prediction.mode.replaceAll('_', ' ')}</p><h2>{prediction.team_a} <span>vs</span> {prediction.team_b}</h2><p className="venue">⌖ {prediction.venue} · {prediction.match_date || 'Neutral-venue simulation'}</p></div>
        <div className={`confidence ${prediction.confidence}`}><span>CONFIDENCE</span><b>{prediction.confidence}</b></div>
      </section>
      <section className="probabilities"><Probability team={prediction.team_a} value={prediction.team_a_win_probability} accent="#38bdf8" /><div className="win-label">WIN PROBABILITY</div><Probability team={prediction.team_b} value={prediction.team_b_win_probability} accent="#fb923c" /></section>

      {h2hData && <H2HSection data={h2hData} />}

      <section className="grid-two">
        <article className="panel explanation"><div className="panel-heading"><span>03</span><div><p>MODEL TRANSPARENCY</p><h2>What moves the odds</h2></div></div>
          {report.model_explanation.map((factor) => {
            const impactScore = (Math.abs(factor.shap_value) * 100).toFixed(1);
            let labelStr = factor.label
              .replace('Team A', prediction.team_a)
              .replace('Team B', prediction.team_b)
              .replace(' A ', ` ${prediction.team_a} `)
              .replace(' B ', ` ${prediction.team_b} `);
            const favorsName = factor.favors === 'Team A' ? prediction.team_a : prediction.team_b;
            return (
              <div className="factor" key={factor.feature}>
                <div><b style={{ textTransform: 'capitalize' }}>{labelStr}</b><small>Favors {favorsName}</small></div>
                <div className="factor-bar"><i className={factor.shap_value >= 0 ? 'positive' : 'negative'} style={{ width: `${Math.min(impactScore, 100)}%` }} /></div>
                <strong className="impact-score">{impactScore} POWER</strong>
              </div>
            );
          })}
        </article>
        <article className="panel preview"><div className="panel-heading"><span>04</span><div><p>GROUND-TRUTHED ANALYSIS</p><h2>Match preview</h2></div></div><blockquote>“{report.grounded_preview.narrative}”</blockquote><p className="note">{report.grounded_preview.disclaimer}</p></article>
      </section>

      <section className="panel players"><div className="panel-heading"><span>05</span><div><p>RECENT IPL FORM</p><h2>Inferred player impact</h2></div></div><p className="note">Derived from each player’s latest 10 IPL appearances. These are not confirmed playing XIs.</p>
        <div className="player-grid">{playerTeams.map(([team, players]) => <div className="player-table" key={team}><h3>{team}</h3><div className="table-head"><span>PLAYER</span><span>RUNS</span><span>WKTS</span><span>IMPACT</span></div>{players.slice(0, 6).map((player) => <div className="player-row" key={player.player}><b>{player.player}</b><span>{player.runs}</span><span>{player.wickets}</span><strong>{player.impact_score}</strong></div>)}</div>)}</div>
      </section>
    </>}

    <section className="panel fixtures">
      <div className="fixture-head">
        <div className="panel-heading"><span>06</span><div><p>FULL SEASON SIMULATION</p><h2>Every match, including playoffs</h2></div></div>
        <label className="season-select">SEASON
          <select value={season} onChange={(event) => setSeason(event.target.value)}>{seasons.map((value) => <option key={value}>{value}</option>)}</select>
        </label>
      </div>
      <div className="fixture-controls">
        <div className="tabs">{[['all', 'ALL MATCHES'], ['league', 'LEAGUE'], ['playoffs', 'PLAYOFFS']].map(([value, label]) =>
          <button className={fixtureFilter === value ? 'active' : ''} onClick={() => setFixtureFilter(value)} key={value}>{label}</button>)}</div>
        <p>{visibleFixtures.length} predictions · Select a match to inspect its model output</p>
      </div>
      <div className="fixture-list">{visibleFixtures.map((fixture) => {
        const aWins = fixture.team_a_win_probability >= fixture.team_b_win_probability
        return <button className={`fixture-card ${selectedFixture?.match_id === fixture.match_id ? 'selected' : ''}`} key={fixture.match_id} onClick={() => setSelectedFixture(fixture)}>
          <small>{fixture.stage} · {fixture.date}</small><div><b>{fixture.team_a}</b><strong>{(fixture.team_a_win_probability * 100).toFixed(0)}%</strong></div>
          <div><b>{fixture.team_b}</b><strong>{(fixture.team_b_win_probability * 100).toFixed(0)}%</strong></div>
          <em>{aWins ? fixture.team_a : fixture.team_b} favoured · {fixture.confidence} data support</em>
        </button>
      })}</div>
      {selectedFixture && <article className="fixture-detail"><p className="eyebrow">SELECTED FIXTURE · {selectedFixture.stage}</p><h3>{selectedFixture.team_a} <span>vs</span> {selectedFixture.team_b}</h3><p>{selectedFixture.venue} · {selectedFixture.date}</p><div className="detail-probabilities"><b>{(selectedFixture.team_a_win_probability * 100).toFixed(1)}% <small>{selectedFixture.team_a}</small></b><i style={{ '--a-win': `${selectedFixture.team_a_win_probability * 100}%` }} /><b>{(selectedFixture.team_b_win_probability * 100).toFixed(1)}% <small>{selectedFixture.team_b}</small></b></div></article>}
    </section>
    <footer>CRICKONOMICS <span>·</span> Local analytics project <span>·</span> Not betting advice <span>·</span> Made with love ❤️ by n ❤️</footer>
  </main>
}

export default App
