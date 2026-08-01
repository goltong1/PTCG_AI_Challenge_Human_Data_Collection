# CABT Railway v5.1.2 Test Report

## Test configuration

- Public mode enabled
- Online matchmaking enabled
- Shared active worker limit: 1
- Test disconnect timeout: 8 seconds
- Test disconnect hint grace: 3 seconds
- Production package defaults: 45 seconds / 15 seconds

## Passed scenarios

1. Quick-match waiter closes the browser
   - Queue before disconnect: 1
   - Queue after grace and cleanup: 0
   - Active workers: 0

2. Two players create one PvP match
   - Active matches: 1
   - Active workers: 1

3. One PvP player closes the browser
   - Remaining player heartbeat triggers cleanup
   - Active matches after grace: 0
   - Active workers after grace: 0

4. Explicit PvP leave
   - Match worker released immediately
   - Active workers: 0

5. AI game browser closes
   - AI worker released after disconnect grace
   - Active workers: 0

6. Browser reload/reconnect within grace
   - Disconnect hint cleared by heartbeat
   - PvP match remains active

7. Browser close signal is lost
   - Missing heartbeat alone expires the AI worker
   - Active workers return to 0

8. Static validation
   - Python modules compile successfully
   - Frontend JavaScript passes `node --check`
