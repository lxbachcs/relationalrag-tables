# BIRD dev schemas for the 11 databases in the audit sample

Source: `dev_tables.json` from BIRD dev_20240627 (table and column names as declared; `original` names are what gold SQL uses). PK = primary key; FK = foreign key target.

## california_schools

### frpm

| column | type | key |
|---|---|---|
| CDSCode | text | PK FK->schools.CDSCode |
| Academic Year | text |  |
| County Code | text |  |
| District Code | integer |  |
| School Code | text |  |
| County Name | text |  |
| District Name | text |  |
| School Name | text |  |
| District Type | text |  |
| School Type | text |  |
| Educational Option Type | text |  |
| NSLP Provision Status | text |  |
| Charter School (Y/N) | integer |  |
| Charter School Number | text |  |
| Charter Funding Type | text |  |
| IRC | integer |  |
| Low Grade | text |  |
| High Grade | text |  |
| Enrollment (K-12) | real |  |
| Free Meal Count (K-12) | real |  |
| Percent (%) Eligible Free (K-12) | real |  |
| FRPM Count (K-12) | real |  |
| Percent (%) Eligible FRPM (K-12) | real |  |
| Enrollment (Ages 5-17) | real |  |
| Free Meal Count (Ages 5-17) | real |  |
| Percent (%) Eligible Free (Ages 5-17) | real |  |
| FRPM Count (Ages 5-17) | real |  |
| Percent (%) Eligible FRPM (Ages 5-17) | real |  |
| 2013-14 CALPADS Fall 1 Certification Status | integer |  |

### satscores

| column | type | key |
|---|---|---|
| cds | text | PK FK->schools.CDSCode |
| rtype | text |  |
| sname | text |  |
| dname | text |  |
| cname | text |  |
| enroll12 | integer |  |
| NumTstTakr | integer |  |
| AvgScrRead | integer |  |
| AvgScrMath | integer |  |
| AvgScrWrite | integer |  |
| NumGE1500 | integer |  |

### schools

| column | type | key |
|---|---|---|
| CDSCode | text | PK |
| NCESDist | text |  |
| NCESSchool | text |  |
| StatusType | text |  |
| County | text |  |
| District | text |  |
| School | text |  |
| Street | text |  |
| StreetAbr | text |  |
| City | text |  |
| Zip | text |  |
| State | text |  |
| MailStreet | text |  |
| MailStrAbr | text |  |
| MailCity | text |  |
| MailZip | text |  |
| MailState | text |  |
| Phone | text |  |
| Ext | text |  |
| Website | text |  |
| OpenDate | date |  |
| ClosedDate | date |  |
| Charter | integer |  |
| CharterNum | text |  |
| FundingType | text |  |
| DOC | text |  |
| DOCType | text |  |
| SOC | text |  |
| SOCType | text |  |
| EdOpsCode | text |  |
| EdOpsName | text |  |
| EILCode | text |  |
| EILName | text |  |
| GSoffered | text |  |
| GSserved | text |  |
| Virtual | text |  |
| Magnet | integer |  |
| Latitude | real |  |
| Longitude | real |  |
| AdmFName1 | text |  |
| AdmLName1 | text |  |
| AdmEmail1 | text |  |
| AdmFName2 | text |  |
| AdmLName2 | text |  |
| AdmEmail2 | text |  |
| AdmFName3 | text |  |
| AdmLName3 | text |  |
| AdmEmail3 | text |  |
| LastUpdate | date |  |

## card_games

### cards

| column | type | key |
|---|---|---|
| id | integer | PK |
| artist | text |  |
| asciiName | text |  |
| availability | text |  |
| borderColor | text |  |
| cardKingdomFoilId | text |  |
| cardKingdomId | text |  |
| colorIdentity | text |  |
| colorIndicator | text |  |
| colors | text |  |
| convertedManaCost | real |  |
| duelDeck | text |  |
| edhrecRank | integer |  |
| faceConvertedManaCost | real |  |
| faceName | text |  |
| flavorName | text |  |
| flavorText | text |  |
| frameEffects | text |  |
| frameVersion | text |  |
| hand | text |  |
| hasAlternativeDeckLimit | integer |  |
| hasContentWarning | integer |  |
| hasFoil | integer |  |
| hasNonFoil | integer |  |
| isAlternative | integer |  |
| isFullArt | integer |  |
| isOnlineOnly | integer |  |
| isOversized | integer |  |
| isPromo | integer |  |
| isReprint | integer |  |
| isReserved | integer |  |
| isStarter | integer |  |
| isStorySpotlight | integer |  |
| isTextless | integer |  |
| isTimeshifted | integer |  |
| keywords | text |  |
| layout | text |  |
| leadershipSkills | text |  |
| life | text |  |
| loyalty | text |  |
| manaCost | text |  |
| mcmId | text |  |
| mcmMetaId | text |  |
| mtgArenaId | text |  |
| mtgjsonV4Id | text |  |
| mtgoFoilId | text |  |
| mtgoId | text |  |
| multiverseId | text |  |
| name | text |  |
| number | text |  |
| originalReleaseDate | text |  |
| originalText | text |  |
| originalType | text |  |
| otherFaceIds | text |  |
| power | text |  |
| printings | text |  |
| promoTypes | text |  |
| purchaseUrls | text |  |
| rarity | text |  |
| scryfallId | text |  |
| scryfallIllustrationId | text |  |
| scryfallOracleId | text |  |
| setCode | text |  |
| side | text |  |
| subtypes | text |  |
| supertypes | text |  |
| tcgplayerProductId | text |  |
| text | text |  |
| toughness | text |  |
| type | text |  |
| types | text |  |
| uuid | text |  |
| variations | text |  |
| watermark | text |  |

### foreign_data

| column | type | key |
|---|---|---|
| id | integer | PK |
| flavorText | text |  |
| language | text |  |
| multiverseid | integer |  |
| name | text |  |
| text | text |  |
| type | text |  |
| uuid | text | FK->cards.uuid |

### legalities

| column | type | key |
|---|---|---|
| id | integer | PK |
| format | text |  |
| status | text |  |
| uuid | text | FK->cards.uuid |

### sets

| column | type | key |
|---|---|---|
| id | integer | PK |
| baseSetSize | integer |  |
| block | text |  |
| booster | text |  |
| code | text |  |
| isFoilOnly | integer |  |
| isForeignOnly | integer |  |
| isNonFoilOnly | integer |  |
| isOnlineOnly | integer |  |
| isPartialPreview | integer |  |
| keyruneCode | text |  |
| mcmId | integer |  |
| mcmIdExtras | integer |  |
| mcmName | text |  |
| mtgoCode | text |  |
| name | text |  |
| parentCode | text |  |
| releaseDate | date |  |
| tcgplayerGroupId | integer |  |
| totalSetSize | integer |  |
| type | text |  |

### set_translations

| column | type | key |
|---|---|---|
| id | integer | PK |
| language | text |  |
| setCode | text | FK->sets.code |
| translation | text |  |

### rulings

| column | type | key |
|---|---|---|
| id | integer | PK |
| date | date |  |
| text | text |  |
| uuid | text | FK->cards.uuid |

## codebase_community

### badges

| column | type | key |
|---|---|---|
| Id | integer | PK |
| UserId | integer | FK->users.Id |
| Name | text |  |
| Date | datetime |  |

### comments

| column | type | key |
|---|---|---|
| Id | integer | PK |
| PostId | integer | FK->posts.Id |
| Score | integer |  |
| Text | text |  |
| CreationDate | datetime |  |
| UserId | integer | FK->users.Id |
| UserDisplayName | text |  |

### postHistory

| column | type | key |
|---|---|---|
| Id | integer | PK |
| PostHistoryTypeId | integer |  |
| PostId | integer | FK->posts.Id |
| RevisionGUID | text |  |
| CreationDate | datetime |  |
| UserId | integer | FK->users.Id |
| Text | text |  |
| Comment | text |  |
| UserDisplayName | text |  |

### postLinks

| column | type | key |
|---|---|---|
| Id | integer | PK |
| CreationDate | datetime |  |
| PostId | integer | FK->posts.Id |
| RelatedPostId | integer | FK->posts.Id |
| LinkTypeId | integer |  |

### posts

| column | type | key |
|---|---|---|
| Id | integer | PK |
| PostTypeId | integer |  |
| AcceptedAnswerId | integer |  |
| CreaionDate | datetime |  |
| Score | integer |  |
| ViewCount | integer |  |
| Body | text |  |
| OwnerUserId | integer | FK->users.Id |
| LasActivityDate | datetime |  |
| Title | text |  |
| Tags | text |  |
| AnswerCount | integer |  |
| CommentCount | integer |  |
| FavoriteCount | integer |  |
| LastEditorUserId | integer | FK->users.Id |
| LastEditDate | datetime |  |
| CommunityOwnedDate | datetime |  |
| ParentId | integer | FK->posts.Id |
| ClosedDate | datetime |  |
| OwnerDisplayName | text |  |
| LastEditorDisplayName | text |  |

### tags

| column | type | key |
|---|---|---|
| Id | integer | PK |
| TagName | text |  |
| Count | integer |  |
| ExcerptPostId | integer | FK->posts.Id |
| WikiPostId | integer |  |

### users

| column | type | key |
|---|---|---|
| Id | integer | PK |
| Reputation | integer |  |
| CreationDate | datetime |  |
| DisplayName | text |  |
| LastAccessDate | datetime |  |
| WebsiteUrl | text |  |
| Location | text |  |
| AboutMe | text |  |
| Views | integer |  |
| UpVotes | integer |  |
| DownVotes | integer |  |
| AccountId | integer |  |
| Age | integer |  |
| ProfileImageUrl | text |  |

### votes

| column | type | key |
|---|---|---|
| Id | integer | PK |
| PostId | integer | FK->posts.Id |
| VoteTypeId | integer |  |
| CreationDate | date |  |
| UserId | integer | FK->users.Id |
| BountyAmount | integer |  |

## debit_card_specializing

### customers

| column | type | key |
|---|---|---|
| CustomerID | integer | PK |
| Segment | text |  |
| Currency | text |  |

### gasstations

| column | type | key |
|---|---|---|
| GasStationID | integer | PK |
| ChainID | integer |  |
| Country | text |  |
| Segment | text |  |

### products

| column | type | key |
|---|---|---|
| ProductID | integer | PK |
| Description | text |  |

### transactions_1k

| column | type | key |
|---|---|---|
| TransactionID | integer | PK |
| Date | date |  |
| Time | text |  |
| CustomerID | integer |  |
| CardID | integer |  |
| GasStationID | integer |  |
| ProductID | integer |  |
| Amount | integer |  |
| Price | real |  |

### yearmonth

| column | type | key |
|---|---|---|
| CustomerID | integer | PK FK->customers.CustomerID |
| Date | text | PK |
| Consumption | real |  |

## european_football_2

### Player_Attributes

| column | type | key |
|---|---|---|
| id | integer | PK |
| player_fifa_api_id | integer | FK->Player.player_fifa_api_id |
| player_api_id | integer | FK->Player.player_api_id |
| date | text |  |
| overall_rating | integer |  |
| potential | integer |  |
| preferred_foot | text |  |
| attacking_work_rate | text |  |
| defensive_work_rate | text |  |
| crossing | integer |  |
| finishing | integer |  |
| heading_accuracy | integer |  |
| short_passing | integer |  |
| volleys | integer |  |
| dribbling | integer |  |
| curve | integer |  |
| free_kick_accuracy | integer |  |
| long_passing | integer |  |
| ball_control | integer |  |
| acceleration | integer |  |
| sprint_speed | integer |  |
| agility | integer |  |
| reactions | integer |  |
| balance | integer |  |
| shot_power | integer |  |
| jumping | integer |  |
| stamina | integer |  |
| strength | integer |  |
| long_shots | integer |  |
| aggression | integer |  |
| interceptions | integer |  |
| positioning | integer |  |
| vision | integer |  |
| penalties | integer |  |
| marking | integer |  |
| standing_tackle | integer |  |
| sliding_tackle | integer |  |
| gk_diving | integer |  |
| gk_handling | integer |  |
| gk_kicking | integer |  |
| gk_positioning | integer |  |
| gk_reflexes | integer |  |

### Player

| column | type | key |
|---|---|---|
| id | integer | PK |
| player_api_id | integer |  |
| player_name | text |  |
| player_fifa_api_id | integer |  |
| birthday | text |  |
| height | integer |  |
| weight | integer |  |

### League

| column | type | key |
|---|---|---|
| id | integer | PK |
| country_id | integer | FK->Country.id |
| name | text |  |

### Country

| column | type | key |
|---|---|---|
| id | integer | PK |
| name | text |  |

### Team

| column | type | key |
|---|---|---|
| id | integer | PK |
| team_api_id | integer |  |
| team_fifa_api_id | integer |  |
| team_long_name | text |  |
| team_short_name | text |  |

### Team_Attributes

| column | type | key |
|---|---|---|
| id | integer | PK |
| team_fifa_api_id | integer | FK->Team.team_fifa_api_id |
| team_api_id | integer | FK->Team.team_api_id |
| date | text |  |
| buildUpPlaySpeed | integer |  |
| buildUpPlaySpeedClass | text |  |
| buildUpPlayDribbling | integer |  |
| buildUpPlayDribblingClass | text |  |
| buildUpPlayPassing | integer |  |
| buildUpPlayPassingClass | text |  |
| buildUpPlayPositioningClass | text |  |
| chanceCreationPassing | integer |  |
| chanceCreationPassingClass | text |  |
| chanceCreationCrossing | integer |  |
| chanceCreationCrossingClass | text |  |
| chanceCreationShooting | integer |  |
| chanceCreationShootingClass | text |  |
| chanceCreationPositioningClass | text |  |
| defencePressure | integer |  |
| defencePressureClass | text |  |
| defenceAggression | integer |  |
| defenceAggressionClass | text |  |
| defenceTeamWidth | integer |  |
| defenceTeamWidthClass | text |  |
| defenceDefenderLineClass | text |  |

### Match

| column | type | key |
|---|---|---|
| id | integer | PK |
| country_id | integer |  |
| league_id | integer |  |
| season | text |  |
| stage | integer |  |
| date | text |  |
| match_api_id | integer |  |
| home_team_api_id | integer | FK->Team.team_api_id |
| away_team_api_id | integer | FK->Team.team_api_id |
| home_team_goal | integer |  |
| away_team_goal | integer |  |
| home_player_X1 | integer |  |
| home_player_X2 | integer |  |
| home_player_X3 | integer |  |
| home_player_X4 | integer |  |
| home_player_X5 | integer |  |
| home_player_X6 | integer |  |
| home_player_X7 | integer |  |
| home_player_X8 | integer |  |
| home_player_X9 | integer |  |
| home_player_X10 | integer |  |
| home_player_X11 | integer |  |
| away_player_X1 | integer |  |
| away_player_X2 | integer |  |
| away_player_X3 | integer |  |
| away_player_X4 | integer |  |
| away_player_X5 | integer |  |
| away_player_X6 | integer |  |
| away_player_X7 | integer |  |
| away_player_X8 | integer |  |
| away_player_X9 | integer |  |
| away_player_X10 | integer |  |
| away_player_X11 | integer |  |
| home_player_Y1 | integer |  |
| home_player_Y2 | integer |  |
| home_player_Y3 | integer |  |
| home_player_Y4 | integer |  |
| home_player_Y5 | integer |  |
| home_player_Y6 | integer |  |
| home_player_Y7 | integer |  |
| home_player_Y8 | integer |  |
| home_player_Y9 | integer |  |
| home_player_Y10 | integer |  |
| home_player_Y11 | integer |  |
| away_player_Y1 | integer |  |
| away_player_Y2 | integer |  |
| away_player_Y3 | integer |  |
| away_player_Y4 | integer |  |
| away_player_Y5 | integer |  |
| away_player_Y6 | integer |  |
| away_player_Y7 | integer |  |
| away_player_Y8 | integer |  |
| away_player_Y9 | integer |  |
| away_player_Y10 | integer |  |
| away_player_Y11 | integer |  |
| home_player_1 | integer | FK->Player.player_api_id |
| home_player_2 | integer | FK->Player.player_api_id |
| home_player_3 | integer | FK->Player.player_api_id |
| home_player_4 | integer | FK->Player.player_api_id |
| home_player_5 | integer | FK->Player.player_api_id |
| home_player_6 | integer | FK->Player.player_api_id |
| home_player_7 | integer | FK->Player.player_api_id |
| home_player_8 | integer | FK->Player.player_api_id |
| home_player_9 | integer | FK->Player.player_api_id |
| home_player_10 | integer | FK->Player.player_api_id |
| home_player_11 | integer | FK->Player.player_api_id |
| away_player_1 | integer | FK->Player.player_api_id |
| away_player_2 | integer | FK->Player.player_api_id |
| away_player_3 | integer | FK->Player.player_api_id |
| away_player_4 | integer | FK->Player.player_api_id |
| away_player_5 | integer | FK->Player.player_api_id |
| away_player_6 | integer | FK->Player.player_api_id |
| away_player_7 | integer | FK->Player.player_api_id |
| away_player_8 | integer | FK->Player.player_api_id |
| away_player_9 | integer | FK->Player.player_api_id |
| away_player_10 | integer | FK->Player.player_api_id |
| away_player_11 | integer | FK->Player.player_api_id |
| goal | text |  |
| shoton | text |  |
| shotoff | text |  |
| foulcommit | text |  |
| card | text |  |
| cross | text |  |
| corner | text |  |
| possession | text |  |
| B365H | real |  |
| B365D | real |  |
| B365A | real |  |
| BWH | real |  |
| BWD | real |  |
| BWA | real |  |
| IWH | real |  |
| IWD | real |  |
| IWA | real |  |
| LBH | real |  |
| LBD | real |  |
| LBA | real |  |
| PSH | real |  |
| PSD | real |  |
| PSA | real |  |
| WHH | real |  |
| WHD | real |  |
| WHA | real |  |
| SJH | real |  |
| SJD | real |  |
| SJA | real |  |
| VCH | real |  |
| VCD | real |  |
| VCA | real |  |
| GBH | real |  |
| GBD | real |  |
| GBA | real |  |
| BSH | real |  |
| BSD | real |  |
| BSA | real |  |

## financial

### account

| column | type | key |
|---|---|---|
| account_id | integer | PK |
| district_id | integer | FK->district.district_id |
| frequency | text |  |
| date | date |  |

### card

| column | type | key |
|---|---|---|
| card_id | integer | PK |
| disp_id | integer | FK->disp.disp_id |
| type | text |  |
| issued | date |  |

### client

| column | type | key |
|---|---|---|
| client_id | integer | PK |
| gender | text |  |
| birth_date | date |  |
| district_id | integer | FK->district.district_id |

### disp

| column | type | key |
|---|---|---|
| disp_id | integer | PK |
| client_id | integer | FK->client.client_id |
| account_id | integer | FK->account.account_id |
| type | text |  |

### district

| column | type | key |
|---|---|---|
| district_id | integer | PK |
| A2 | text |  |
| A3 | text |  |
| A4 | text |  |
| A5 | text |  |
| A6 | text |  |
| A7 | text |  |
| A8 | integer |  |
| A9 | integer |  |
| A10 | real |  |
| A11 | integer |  |
| A12 | real |  |
| A13 | real |  |
| A14 | integer |  |
| A15 | integer |  |
| A16 | integer |  |

### loan

| column | type | key |
|---|---|---|
| loan_id | integer | PK |
| account_id | integer | FK->account.account_id |
| date | date |  |
| amount | integer |  |
| duration | integer |  |
| payments | real |  |
| status | text |  |

### order

| column | type | key |
|---|---|---|
| order_id | integer | PK |
| account_id | integer | FK->account.account_id |
| bank_to | text |  |
| account_to | integer |  |
| amount | real |  |
| k_symbol | text |  |

### trans

| column | type | key |
|---|---|---|
| trans_id | integer | PK |
| account_id | integer | FK->account.account_id |
| date | date |  |
| type | text |  |
| operation | text |  |
| amount | integer |  |
| balance | integer |  |
| k_symbol | text |  |
| bank | text |  |
| account | integer |  |

## formula_1

### circuits

| column | type | key |
|---|---|---|
| circuitId | integer | PK |
| circuitRef | text |  |
| name | text |  |
| location | text |  |
| country | text |  |
| lat | real |  |
| lng | real |  |
| alt | integer |  |
| url | text |  |

### constructors

| column | type | key |
|---|---|---|
| constructorId | integer | PK |
| constructorRef | text |  |
| name | text |  |
| nationality | text |  |
| url | text |  |

### drivers

| column | type | key |
|---|---|---|
| driverId | integer | PK |
| driverRef | text |  |
| number | integer |  |
| code | text |  |
| forename | text |  |
| surname | text |  |
| dob | date |  |
| nationality | text |  |
| url | text |  |

### seasons

| column | type | key |
|---|---|---|
| year | integer | PK |
| url | text |  |

### races

| column | type | key |
|---|---|---|
| raceId | integer | PK |
| year | integer | FK->seasons.year |
| round | integer |  |
| circuitId | integer | FK->circuits.circuitId |
| name | text |  |
| date | date |  |
| time | text |  |
| url | text |  |

### constructorResults

| column | type | key |
|---|---|---|
| constructorResultsId | integer | PK |
| raceId | integer | FK->races.raceId |
| constructorId | integer | FK->constructors.constructorId |
| points | real |  |
| status | text |  |

### constructorStandings

| column | type | key |
|---|---|---|
| constructorStandingsId | integer | PK |
| raceId | integer | FK->races.raceId |
| constructorId | integer | FK->constructors.constructorId |
| points | real |  |
| position | integer |  |
| positionText | text |  |
| wins | integer |  |

### driverStandings

| column | type | key |
|---|---|---|
| driverStandingsId | integer | PK |
| raceId | integer | FK->races.raceId |
| driverId | integer | FK->drivers.driverId |
| points | real |  |
| position | integer |  |
| positionText | text |  |
| wins | integer |  |

### lapTimes

| column | type | key |
|---|---|---|
| raceId | integer | PK FK->races.raceId |
| driverId | integer | PK FK->drivers.driverId |
| lap | integer | PK |
| position | integer |  |
| time | text |  |
| milliseconds | integer |  |

### pitStops

| column | type | key |
|---|---|---|
| raceId | integer | PK FK->races.raceId |
| driverId | integer | PK FK->drivers.driverId |
| stop | integer | PK |
| lap | integer |  |
| time | text |  |
| duration | text |  |
| milliseconds | integer |  |

### qualifying

| column | type | key |
|---|---|---|
| qualifyId | integer | PK |
| raceId | integer | FK->races.raceId |
| driverId | integer | FK->drivers.driverId |
| constructorId | integer | FK->constructors.constructorId |
| number | integer |  |
| position | integer |  |
| q1 | text |  |
| q2 | text |  |
| q3 | text |  |

### status

| column | type | key |
|---|---|---|
| statusId | integer | PK |
| status | text |  |

### results

| column | type | key |
|---|---|---|
| resultId | integer | PK |
| raceId | integer | FK->races.raceId |
| driverId | integer | FK->drivers.driverId |
| constructorId | integer | FK->constructors.constructorId |
| number | integer |  |
| grid | integer |  |
| position | integer |  |
| positionText | text |  |
| positionOrder | integer |  |
| points | real |  |
| laps | integer |  |
| time | text |  |
| milliseconds | integer |  |
| fastestLap | integer |  |
| rank | integer |  |
| fastestLapTime | text |  |
| fastestLapSpeed | text |  |
| statusId | integer | FK->status.statusId |

## student_club

### event

| column | type | key |
|---|---|---|
| event_id | text | PK |
| event_name | text |  |
| event_date | text |  |
| type | text |  |
| notes | text |  |
| location | text |  |
| status | text |  |

### major

| column | type | key |
|---|---|---|
| major_id | text | PK |
| major_name | text |  |
| department | text |  |
| college | text |  |

### zip_code

| column | type | key |
|---|---|---|
| zip_code | integer | PK |
| type | text |  |
| city | text |  |
| county | text |  |
| state | text |  |
| short_state | text |  |

### attendance

| column | type | key |
|---|---|---|
| link_to_event | text | PK FK->event.event_id |
| link_to_member | text | PK FK->member.member_id |

### budget

| column | type | key |
|---|---|---|
| budget_id | text | PK |
| category | text |  |
| spent | real |  |
| remaining | real |  |
| amount | integer |  |
| event_status | text |  |
| link_to_event | text | FK->event.event_id |

### expense

| column | type | key |
|---|---|---|
| expense_id | text | PK |
| expense_description | text |  |
| expense_date | text |  |
| cost | real |  |
| approved | text |  |
| link_to_member | text | FK->member.member_id |
| link_to_budget | text | FK->budget.budget_id |

### income

| column | type | key |
|---|---|---|
| income_id | text | PK |
| date_received | text |  |
| amount | integer |  |
| source | text |  |
| notes | text |  |
| link_to_member | text | FK->member.member_id |

### member

| column | type | key |
|---|---|---|
| member_id | text | PK |
| first_name | text |  |
| last_name | text |  |
| email | text |  |
| position | text |  |
| t_shirt_size | text |  |
| phone | text |  |
| zip | integer | FK->zip_code.zip_code |
| link_to_major | text | FK->major.major_id |

## superhero

### alignment

| column | type | key |
|---|---|---|
| id | integer | PK |
| alignment | text |  |

### attribute

| column | type | key |
|---|---|---|
| id | integer | PK |
| attribute_name | text |  |

### colour

| column | type | key |
|---|---|---|
| id | integer | PK |
| colour | text |  |

### gender

| column | type | key |
|---|---|---|
| id | integer | PK |
| gender | text |  |

### publisher

| column | type | key |
|---|---|---|
| id | integer | PK |
| publisher_name | text |  |

### race

| column | type | key |
|---|---|---|
| id | integer | PK |
| race | text |  |

### superhero

| column | type | key |
|---|---|---|
| id | integer | PK |
| superhero_name | text |  |
| full_name | text |  |
| gender_id | integer | FK->gender.id |
| eye_colour_id | integer | FK->colour.id |
| hair_colour_id | integer | FK->colour.id |
| skin_colour_id | integer | FK->colour.id |
| race_id | integer | FK->race.id |
| publisher_id | integer | FK->publisher.id |
| alignment_id | integer | FK->alignment.id |
| height_cm | integer |  |
| weight_kg | integer |  |

### hero_attribute

| column | type | key |
|---|---|---|
| hero_id | integer | FK->superhero.id |
| attribute_id | integer | FK->attribute.id |
| attribute_value | integer |  |

### superpower

| column | type | key |
|---|---|---|
| id | integer | PK |
| power_name | text |  |

### hero_power

| column | type | key |
|---|---|---|
| hero_id | integer | FK->superhero.id |
| power_id | integer | FK->superpower.id |

## thrombosis_prediction

### Examination

| column | type | key |
|---|---|---|
| ID | integer | FK->Patient.ID |
| Examination Date | date |  |
| aCL IgG | real |  |
| aCL IgM | real |  |
| ANA | integer |  |
| ANA Pattern | text |  |
| aCL IgA | integer |  |
| Diagnosis | text |  |
| KCT | text |  |
| RVVT | text |  |
| LAC | text |  |
| Symptoms | text |  |
| Thrombosis | integer |  |

### Patient

| column | type | key |
|---|---|---|
| ID | integer | PK |
| SEX | text |  |
| Birthday | date |  |
| Description | date |  |
| First Date | date |  |
| Admission | text |  |
| Diagnosis | text |  |

### Laboratory

| column | type | key |
|---|---|---|
| ID | integer | PK FK->Patient.ID |
| Date | date | PK |
| GOT | integer |  |
| GPT | integer |  |
| LDH | integer |  |
| ALP | integer |  |
| TP | real |  |
| ALB | real |  |
| UA | real |  |
| UN | integer |  |
| CRE | real |  |
| T-BIL | real |  |
| T-CHO | integer |  |
| TG | integer |  |
| CPK | integer |  |
| GLU | integer |  |
| WBC | real |  |
| RBC | real |  |
| HGB | real |  |
| HCT | real |  |
| PLT | integer |  |
| PT | real |  |
| APTT | integer |  |
| FG | real |  |
| PIC | integer |  |
| TAT | integer |  |
| TAT2 | integer |  |
| U-PRO | text |  |
| IGG | integer |  |
| IGA | integer |  |
| IGM | integer |  |
| CRP | text |  |
| RA | text |  |
| RF | text |  |
| C3 | integer |  |
| C4 | integer |  |
| RNP | text |  |
| SM | text |  |
| SC170 | text |  |
| SSA | text |  |
| SSB | text |  |
| CENTROMEA | text |  |
| DNA | text |  |
| DNA-II | integer |  |

## toxicology

### atom

| column | type | key |
|---|---|---|
| atom_id | text | PK |
| molecule_id | text | FK->molecule.molecule_id |
| element | text |  |

### bond

| column | type | key |
|---|---|---|
| bond_id | text | PK |
| molecule_id | text | FK->molecule.molecule_id |
| bond_type | text |  |

### connected

| column | type | key |
|---|---|---|
| atom_id | text | PK FK->atom.atom_id |
| atom_id2 | text | PK FK->atom.atom_id |
| bond_id | text | FK->bond.bond_id |

### molecule

| column | type | key |
|---|---|---|
| molecule_id | text | PK |
| label | text |  |
