# PHASE 1: FOUNDATION - Детальный план реализации

**Период:** 3-6 месяцев (February - July 2026)
**Цель:** Подготовка базовой инфраструктуры для Autonomous Capital Civilization
**Бюджет:** $50K-$100K (в зависимости от команды)
**Команда:** 2-4 разработчика + you

---

## 📋 OVERVIEW

Phase 1 закладывает фундамент для трансформации HEAN в глобальную платформу:

1. **Risk Constitution** — неизменяемые правила безопасности
2. **Transparency Engine** — полная прозрачность решений
3. **Multi-User Infrastructure** — поддержка тысяч пользователей
4. **Cloud Deployment** — масштабируемая инфраструктура

---

## 🎯 DELIVERABLES

### Must-Have (Критично)

- ✅ Risk Constitution smart contract (deployed на Ethereum)
- ✅ Multi-tenant backend (100+ users)
- ✅ Transparency dashboard (public)
- ✅ Kubernetes deployment (production-ready)
- ✅ Billing system (Stripe integration)
- ✅ API documentation (OpenAPI)

### Nice-to-Have (Желательно)

- ⭐ Mobile-responsive UI
- ⭐ Email notifications
- ⭐ Telegram bot
- ⭐ Community Discord

---

## 📅 DETAILED TIMELINE

### Month 1: Planning & Architecture

#### Week 1-2: Requirements & Design

**Tasks:**
1. **Constitutional Smart Contract Specification**
   - [ ] Define all risk parameters (drawdown, leverage, position size)
   - [ ] Design governance mechanism (voting, timelock)
   - [ ] Plan kill switch hierarchy
   - [ ] Security review process

2. **Database Architecture**
   - [ ] Design multi-tenant schema
   - [ ] Plan data partitioning strategy
   - [ ] Define indexes for performance
   - [ ] Migration plan from single-user

3. **Infrastructure Planning**
   - [ ] Kubernetes cluster design
   - [ ] Service topology (API, Worker, DB, Redis, Kafka)
   - [ ] Auto-scaling rules
   - [ ] Disaster recovery plan

4. **Team Setup**
   - [ ] Hire/contract developers
   - [ ] Set up development environments
   - [ ] Code review process
   - [ ] Communication channels (Slack, Notion)

**Deliverables:**
- Technical design document
- Database schema v1
- Kubernetes architecture diagram
- Team onboarding complete

---

#### Week 3-4: Development Environment Setup

**Tasks:**
1. **Development Infrastructure**
   ```bash
   # Set up local Kubernetes (minikube or k3s)
   minikube start --cpus=4 --memory=8192

   # Install Helm
   helm repo add bitnami https://charts.bitnami.com/bitnami

   # Deploy PostgreSQL
   helm install postgresql bitnami/postgresql

   # Deploy Redis
   helm install redis bitnami/redis

   # Deploy Kafka
   helm install kafka bitnami/kafka
   ```

2. **CI/CD Pipeline**
   - [ ] GitHub Actions workflows
   - [ ] Docker image building
   - [ ] Automated testing
   - [ ] Deployment automation

3. **Monitoring Stack**
   ```bash
   # Install Prometheus
   helm install prometheus prometheus-community/prometheus

   # Install Grafana
   helm install grafana grafana/grafana

   # Install Loki (logging)
   helm install loki grafana/loki-stack
   ```

4. **Security Setup**
   - [ ] HashiCorp Vault for secrets
   - [ ] SSL certificates (Let's Encrypt)
   - [ ] Firewall rules
   - [ ] DDoS protection (Cloudflare)

**Deliverables:**
- Working dev environment
- CI/CD pipeline operational
- Monitoring dashboards
- Security baseline established

---

### Month 2: Risk Constitution Layer

#### Week 5-6: Smart Contract Development

**File:** `contracts/RiskConstitution.sol`

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/governance/TimelockController.sol";

/**
 * @title RiskConstitution
 * @dev Immutable risk parameters enforced on-chain
 */
contract RiskConstitution is AccessControl {
    bytes32 public constant GOVERNOR_ROLE = keccak256("GOVERNOR_ROLE");
    bytes32 public constant AUDITOR_ROLE = keccak256("AUDITOR_ROLE");

    // ARTICLE I: Capital Preservation
    uint256 public constant MAX_DRAWDOWN_BPS = 1500;  // 15%
    uint256 public constant MAX_POSITION_SIZE_BPS = 500;  // 5%
    uint256 public constant MAX_LEVERAGE = 3;  // 3x

    // ARTICLE II: Risk Limits
    uint256 public constant MAX_DAILY_LOSS_BPS = 300;  // 3%
    uint256 public constant MAX_CORRELATION_BPS = 7000;  // 0.7
    uint256 public constant MIN_LIQUIDITY_BUFFER_BPS = 2000;  // 20%

    // ARTICLE III: Circuit Breakers
    mapping(address => bool) public killSwitchActive;
    mapping(address => uint256) public lastKillSwitchTime;

    // ARTICLE IV: Governance
    TimelockController public timelock;
    uint256 public constant PROPOSAL_TIMELOCK = 30 days;
    uint256 public constant MIN_VOTES_FOR_CHANGE = 9000;  // 90%

    event KillSwitchActivated(address indexed agent, string reason);
    event ConstitutionProposed(bytes32 indexed proposalId, string description);
    event ConstitutionUpdated(bytes32 indexed proposalId);

    constructor(address[] memory governors) {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);

        for (uint i = 0; i < governors.length; i++) {
            _grantRole(GOVERNOR_ROLE, governors[i]);
        }

        // Deploy timelock
        address[] memory proposers = new address[](1);
        proposers[0] = address(this);
        address[] memory executors = new address[](1);
        executors[0] = address(0);  // Anyone can execute

        timelock = new TimelockController(
            PROPOSAL_TIMELOCK,
            proposers,
            executors,
            msg.sender
        );
    }

    /**
     * @dev Activate kill switch for an agent
     * @param agent Address of the agent
     * @param reason Reason for activation
     */
    function activateKillSwitch(
        address agent,
        string memory reason
    ) external onlyRole(GOVERNOR_ROLE) {
        require(!killSwitchActive[agent], "Already active");

        killSwitchActive[agent] = true;
        lastKillSwitchTime[agent] = block.timestamp;

        emit KillSwitchActivated(agent, reason);
    }

    /**
     * @dev Check if position size is within limits
     */
    function checkPositionSize(
        uint256 positionValue,
        uint256 portfolioValue
    ) external pure returns (bool) {
        uint256 positionSizeBps = (positionValue * 10000) / portfolioValue;
        return positionSizeBps <= MAX_POSITION_SIZE_BPS;
    }

    /**
     * @dev Check if drawdown is within limits
     */
    function checkDrawdown(
        uint256 currentEquity,
        uint256 peakEquity
    ) external pure returns (bool) {
        if (peakEquity == 0) return true;

        uint256 drawdownBps = ((peakEquity - currentEquity) * 10000) / peakEquity;
        return drawdownBps <= MAX_DRAWDOWN_BPS;
    }

    /**
     * @dev Propose constitution change (requires governance)
     */
    function proposeChange(
        string memory description,
        bytes memory data
    ) external onlyRole(GOVERNOR_ROLE) returns (bytes32) {
        bytes32 proposalId = keccak256(abi.encodePacked(description, data, block.timestamp));

        emit ConstitutionProposed(proposalId, description);

        // Queue in timelock
        timelock.schedule(
            address(this),
            0,
            data,
            bytes32(0),
            proposalId,
            PROPOSAL_TIMELOCK
        );

        return proposalId;
    }
}
```

**Tasks:**
- [ ] Write smart contract
- [ ] Unit tests (100% coverage)
- [ ] Security audit (Trail of Bits or OpenZeppelin)
- [ ] Deploy to testnet (Sepolia)
- [ ] Deploy to mainnet (Ethereum)

**Budget:** $10K-$20K (audit)

---

#### Week 7-8: HSM Integration & Kill Switch

**File:** `src/hean/constitution/enforcer.py`

```python
import hashlib
import hmac
from typing import Dict, Optional
from web3 import Web3
from eth_account import Account

class ConstitutionEnforcer:
    """
    Enforces Risk Constitution locally + on-chain

    Two-layer enforcement:
    1. Local checks (fast, <1ms)
    2. On-chain verification (slower, but immutable proof)
    """

    def __init__(
        self,
        web3: Web3,
        contract_address: str,
        hsm_client: HSMClient,
    ):
        self.web3 = web3
        self.contract = web3.eth.contract(
            address=contract_address,
            abi=RISK_CONSTITUTION_ABI,
        )
        self.hsm = hsm_client

    def check_position_size(
        self,
        position_value: float,
        portfolio_value: float,
    ) -> bool:
        """
        Check position size limit (local)
        """
        if portfolio_value == 0:
            return False

        position_size_pct = position_value / portfolio_value

        # Local check (fast)
        if position_size_pct > 0.05:  # 5% limit
            return False

        # On-chain verification (proof)
        is_valid = self.contract.functions.checkPositionSize(
            int(position_value * 1e18),  # Convert to wei
            int(portfolio_value * 1e18),
        ).call()

        return is_valid

    def check_drawdown(
        self,
        current_equity: float,
        peak_equity: float,
    ) -> bool:
        """
        Check drawdown limit (local + on-chain)
        """
        if peak_equity == 0:
            return True

        drawdown_pct = (peak_equity - current_equity) / peak_equity

        # Local check
        if drawdown_pct > 0.15:  # 15% limit
            return False

        # On-chain verification
        is_valid = self.contract.functions.checkDrawdown(
            int(current_equity * 1e18),
            int(peak_equity * 1e18),
        ).call()

        return is_valid

    def activate_kill_switch(
        self,
        agent_id: str,
        reason: str,
    ):
        """
        Activate kill switch (requires HSM signature)
        """
        # Build transaction
        tx = self.contract.functions.activateKillSwitch(
            agent_id,
            reason,
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'gas': 200000,
            'gasPrice': self.web3.eth.gas_price,
        })

        # Sign with HSM (secure)
        signed_tx = self.hsm.sign_transaction(tx)

        # Send transaction
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)

        # Wait for confirmation
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

        return receipt
```

**Tasks:**
- [ ] Integrate Web3.py
- [ ] HSM client implementation
- [ ] Kill switch hierarchy (agent → portfolio → system)
- [ ] Real-time monitoring dashboard
- [ ] Alert system (email, Telegram, PagerDuty)

---

### Month 3: Transparency Engine

#### Week 9-10: Decision Logging System

**File:** `src/hean/transparency/decision_log.py`

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib
import json

@dataclass
class DecisionLog:
    """
    Immutable log entry for every decision made by system
    """
    timestamp: int  # nanoseconds since epoch
    agent_id: str
    decision_type: str  # "entry", "exit", "rebalance", "kill"

    # Input state
    market_state: Dict[str, Any]
    portfolio_state: Dict[str, Any]

    # Decision rationale
    rationale: str
    confidence: float

    # Risk checks
    risk_checks: Dict[str, bool]
    constitution_compliance: bool

    # Execution
    execution_details: Optional[Dict[str, Any]] = None

    # Outcome (filled after execution)
    realized_pnl: Optional[float] = None
    slippage: Optional[float] = None

    def hash(self) -> str:
        """
        Calculate content hash for Merkle tree
        """
        content = json.dumps({
            'timestamp': self.timestamp,
            'agent_id': self.agent_id,
            'decision_type': self.decision_type,
            'market_state': self.market_state,
            'portfolio_state': self.portfolio_state,
            'rationale': self.rationale,
            'risk_checks': self.risk_checks,
            'execution_details': self.execution_details,
        }, sort_keys=True)

        return hashlib.sha256(content.encode()).hexdigest()


class DecisionLogger:
    """
    Logs all decisions with Merkle tree for blockchain anchoring
    """

    def __init__(self, db, blockchain):
        self.db = db
        self.blockchain = blockchain
        self.merkle_tree = []
        self.block_size = 1000  # Anchor every 1000 decisions

    async def log_decision(self, decision: DecisionLog):
        """
        Log decision locally + add to Merkle tree
        """
        # Store in database (fast, local)
        await self.db.insert_decision(decision)

        # Add to Merkle tree
        decision_hash = decision.hash()
        self.merkle_tree.append(decision_hash)

        # Anchor to blockchain if block is full
        if len(self.merkle_tree) >= self.block_size:
            await self.anchor_to_blockchain()

    async def anchor_to_blockchain(self):
        """
        Anchor Merkle root to Ethereum blockchain
        """
        # Calculate Merkle root
        merkle_root = self.calculate_merkle_root(self.merkle_tree)

        # Store on blockchain
        tx_hash = await self.blockchain.anchor_merkle_root(merkle_root)

        # Store anchor reference
        await self.db.store_anchor(
            merkle_root=merkle_root,
            tx_hash=tx_hash,
            decision_hashes=self.merkle_tree,
        )

        # Reset tree
        self.merkle_tree = []

    def calculate_merkle_root(self, leaves):
        """
        Calculate Merkle root from leaves
        """
        if len(leaves) == 0:
            return "0" * 64
        if len(leaves) == 1:
            return leaves[0]

        # Build tree bottom-up
        tree = leaves[:]
        while len(tree) > 1:
            if len(tree) % 2 == 1:
                tree.append(tree[-1])  # Duplicate last element

            tree = [
                hashlib.sha256((tree[i] + tree[i+1]).encode()).hexdigest()
                for i in range(0, len(tree), 2)
            ]

        return tree[0]
```

**Tasks:**
- [ ] Decision logging system
- [ ] Merkle tree implementation
- [ ] Blockchain anchoring (every 1000 decisions)
- [ ] Proof generation & verification
- [ ] Database optimization (indexes, partitioning)

---

#### Week 11-12: Public Transparency Dashboard

**File:** `apps/dashboard/src/pages/Transparency.tsx`

```typescript
import React, { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';

interface Decision {
  timestamp: number;
  agentId: string;
  decisionType: string;
  rationale: string;
  riskChecks: Record<string, boolean>;
  constitutionCompliance: boolean;
  realizedPnl?: number;
}

export default function TransparencyDashboard() {
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [merkleAnchors, setMerkleAnchors] = useState([]);
  const [stats, setStats] = useState({
    totalDecisions: 0,
    successRate: 0,
    avgPnl: 0,
    constitutionViolations: 0,
  });

  useEffect(() => {
    // Fetch real-time decisions
    const ws = new WebSocket('wss://api.hean.ai/transparency/stream');
    ws.onmessage = (event) => {
      const decision = JSON.parse(event.data);
      setDecisions(prev => [decision, ...prev].slice(0, 100));
    };

    // Fetch stats
    fetch('/api/transparency/stats').then(r => r.json()).then(setStats);

    // Fetch Merkle anchors
    fetch('/api/transparency/anchors').then(r => r.json()).then(setMerkleAnchors);

    return () => ws.close();
  }, []);

  return (
    <div className="transparency-dashboard">
      <h1>Transparency Dashboard</h1>
      <p>Every decision is logged and anchored to Ethereum blockchain</p>

      {/* Stats Overview */}
      <div className="stats-grid">
        <StatCard title="Total Decisions" value={stats.totalDecisions.toLocaleString()} />
        <StatCard title="Success Rate" value={`${stats.successRate.toFixed(1)}%`} />
        <StatCard title="Avg PnL" value={`$${stats.avgPnl.toFixed(2)}`} />
        <StatCard title="Constitution Violations" value={stats.constitutionViolations} color="red" />
      </div>

      {/* Real-time Decision Stream */}
      <section>
        <h2>Live Decision Stream</h2>
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Agent</th>
              <th>Type</th>
              <th>Rationale</th>
              <th>Constitution</th>
              <th>PnL</th>
            </tr>
          </thead>
          <tbody>
            {decisions.map((d, i) => (
              <tr key={i}>
                <td>{new Date(d.timestamp / 1000000).toLocaleString()}</td>
                <td>{d.agentId.slice(0, 8)}</td>
                <td>{d.decisionType}</td>
                <td>{d.rationale}</td>
                <td>{d.constitutionCompliance ? '✅' : '❌'}</td>
                <td className={d.realizedPnl > 0 ? 'green' : 'red'}>
                  {d.realizedPnl !== undefined ? `$${d.realizedPnl.toFixed(2)}` : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* Blockchain Anchors */}
      <section>
        <h2>Blockchain Anchors</h2>
        <p>Merkle roots anchored to Ethereum for immutability</p>
        <table>
          <thead>
            <tr>
              <th>Block</th>
              <th>Merkle Root</th>
              <th>Tx Hash</th>
              <th>Decisions</th>
              <th>Verify</th>
            </tr>
          </thead>
          <tbody>
            {merkleAnchors.map((anchor, i) => (
              <tr key={i}>
                <td>{anchor.blockNumber}</td>
                <td><code>{anchor.merkleRoot.slice(0, 16)}...</code></td>
                <td>
                  <a href={`https://etherscan.io/tx/${anchor.txHash}`} target="_blank">
                    {anchor.txHash.slice(0, 16)}...
                  </a>
                </td>
                <td>{anchor.decisionCount}</td>
                <td>
                  <button onClick={() => verifyAnchor(anchor)}>Verify</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
```

**Tasks:**
- [ ] React dashboard (real-time updates)
- [ ] WebSocket streaming (live decisions)
- [ ] Blockchain verification UI
- [ ] Export audit reports (PDF, CSV)
- [ ] Mobile-responsive design

---

### Month 4-5: Multi-User Infrastructure

*(Продолжение в следующей части из-за лимита длины)*

---

## 💰 БЮДЖЕТ ФАЗЫ 1

| Категория | Описание | Стоимость |
|-----------|----------|-----------|
| **Development** | 2-4 разработчика × 6 месяцев | $30K-$60K |
| **Security Audit** | Smart contract audit (Trail of Bits) | $10K-$20K |
| **Infrastructure** | AWS/GCP (development + production) | $2K-$5K |
| **Tools & Services** | GitHub, Sentry, monitoring, APIs | $1K-$3K |
| **Legal** | Entity formation, terms of service | $2K-$5K |
| **Marketing** | Landing page, content creation | $2K-$5K |
| **Miscellaneous** | Buffer for unexpected costs | $3K-$7K |
| **TOTAL** | | **$50K-$105K** |

---

## ✅ SUCCESS CRITERIA

### Technical KPIs

- ✅ Smart contract deployed (mainnet)
- ✅ 99.9% uptime (production)
- ✅ <100ms API latency (p95)
- ✅ 100+ users supported (load tested)
- ✅ Zero security vulnerabilities (critical/high)

### Business KPIs

- ✅ 100 beta users signed up
- ✅ $5K MRR (subscriptions)
- ✅ 90% user satisfaction (NPS > 50)
- ✅ 10+ institutional inquiries

### Product KPIs

- ✅ All Constitutional limits enforced
- ✅ 1000+ decisions logged daily
- ✅ Public dashboard live (1000+ visitors)
- ✅ Zero Constitutional violations

---

## 🚨 RISKS & MITIGATION

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Smart contract vulnerability | Critical | Medium | Professional audit, bug bounty, timelock |
| Scaling issues (>1000 users) | High | Medium | Load testing, horizontal scaling, CDN |
| Regulatory issues | High | Low | Legal counsel, compliance-first design |
| Competition (copycats) | Medium | High | First-mover advantage, network effects, brand |
| Team attrition | Medium | Low | Clear vision, equity, remote-friendly |

---

## 📞 SUPPORT & RESOURCES

### Technical Resources

- **Smart Contract Templates:** OpenZeppelin
- **Audit Services:** Trail of Bits, Certik, OpenZeppelin
- **Infrastructure:** AWS, GCP, DigitalOcean
- **Monitoring:** Grafana, Prometheus, Sentry

### Community Resources

- **Discord:** HEAN Community
- **GitHub:** github.com/hean-ai
- **Documentation:** docs.hean.ai
- **Blog:** blog.hean.ai

---

## 📝 NEXT STEPS

1. **Review this plan** with your team
2. **Finalize budget** and timeline
3. **Hire developers** (if needed)
4. **Start Week 1** (requirements & design)

**Let's build the future of autonomous capital! 🚀**
